#!/usr/bin/env python3
"""Export real trace.npz recordings to a self-contained, offline 3D replay."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.core.geometry import (
    COATING_SPHERE_CENTERS_BODY_M, COATING_SPHERE_RADIUS_M,
    PALM_CENTER_OFFSETS_BODY_M, PALM_NORMAL_AXES_BODY,
    PALM_CORE_CENTERS_BODY_M, PALM_CORE_SIZE_M,
)
from moz1_catch.core.prediction import BoxFlight
from moz1_catch.kinematics import named_angles, palms_from_angles, parse_joints

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS = ROOT / 'output'
DEFAULT_MESHES = ROOT / 'data/meshes'


def numbers(value):
    """Six decimal places: sub-micrometre position, <0.001 degree angle precision."""
    return np.asarray(value).round(6).tolist()


def packed(array, dtype):
    raw = array if isinstance(array, bytes) else np.asarray(array, dtype=dtype).tobytes()
    return base64.b64encode(gzip.compress(raw,
                                         mtime=0)).decode('ascii')


def stl_geometry(path):
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) != 84 + 50 * count or count == 0:
        raise ValueError(f'Expected a non-empty binary STL: {path}')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)),
                      ('attribute', '<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84, count=count)['vertices']
    if not np.isfinite(triangles).all():
        raise ValueError(f'Non-finite STL vertices: {path}')
    # Exact welding only: no triangle sampling, no holes or changes to mesh shape.
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return dict(kind='mesh', vertices=packed(vertices, '<f4'), indices=packed(indices, '<u4'),
                triangles=count)


def origin_pose(element):
    xyz = [float(x) for x in element.get('xyz', '0 0 0').split()] if element is not None else [0]*3
    rpy = [float(x) for x in element.get('rpy', '0 0 0').split()] if element is not None else [0]*3
    return numbers(xyz + Rotation.from_euler('xyz', rpy).as_quat().tolist())


def robot_model(urdf, meshes):
    root = ET.parse(urdf).getroot()
    model = dict(links=[link.get('name') for link in root.findall('link')],
                 joints=[], visuals=[], geometries={})
    for name, joint in parse_joints(urdf).items():
        model['joints'].append(dict(name=name, parent=joint['parent'], child=joint['child'],
                                    type=joint['type'], axis=numbers(joint['axis'])
                                    if joint['axis'] is not None else [0, 0, 1],
                                    pose=numbers(np.r_[joint['T_origin'][:3, 3],
                                        Rotation.from_matrix(joint['T_origin'][:3, :3]).as_quat()])))
    for link in root.findall('link'):
        # The runtime replaces the URDF's old six-sphere hand collision proxies.
        if link.get('name') in ('left_rubber_hand', 'right_rubber_hand'):
            continue
        for visual in link.findall('visual'):
            shape = next(iter(visual.find('geometry')))
            kind = shape.tag
            if kind == 'mesh':
                name = Path(shape.get('filename')).name
                path = meshes / name
                if not path.is_file():
                    raise FileNotFoundError(f'{path}: missing repository asset; use --meshes for another mesh directory')
                if name not in model['geometries']:
                    model['geometries'][name] = stl_geometry(path)
                geometry = dict(kind=kind, name=name,
                                scale=[float(x) for x in shape.get('scale', '1 1 1').split()])
            elif kind in ('sphere', 'cylinder', 'box'):
                geometry = dict(kind=kind, **{k: [float(x) for x in v.split()] if k == 'size'
                                             else float(v) for k, v in shape.attrib.items()})
            else:
                raise ValueError(f'Unsupported URDF visual: {kind}')
            model['visuals'].append(dict(link=link.get('name'), pose=origin_pose(visual.find('origin')),
                                         geometry=geometry))
    for side, link in enumerate(('left_rubber_hand', 'right_rubber_hand')):
        for center in COATING_SPHERE_CENTERS_BODY_M[side]:
            model['visuals'].append(dict(link=link, role='coating',
                pose=numbers((*center, 0, 0, 0, 1)),
                geometry=dict(kind='sphere', radius=COATING_SPHERE_RADIUS_M)))
        model['visuals'].append(dict(link=link, role='core',
            pose=numbers((*PALM_CORE_CENTERS_BODY_M[side], 0, 0, 0, 1)),
            geometry=dict(kind='box', size=numbers(PALM_CORE_SIZE_M))))
    return model


def check_hand_model(model):
    """Catch regressions to the legacy URDF hand or a shifted palm frame."""
    for side, link in enumerate(('left_rubber_hand', 'right_rubber_hand')):
        shapes = [v for v in model['visuals'] if v['link'] == link]
        coating = [v for v in shapes if v.get('role') == 'coating']
        core = [v for v in shapes if v.get('role') == 'core']
        assert len(shapes) == 13 and len(coating) == 12 and len(core) == 1
        np.testing.assert_allclose([v['pose'][:3] for v in coating],
                                   COATING_SPHERE_CENTERS_BODY_M[side], atol=1e-9)
        assert all(v['geometry']['radius'] == COATING_SPHERE_RADIUS_M for v in coating)
        np.testing.assert_allclose(core[0]['pose'][:3], PALM_CORE_CENTERS_BODY_M[side], atol=1e-9)
        np.testing.assert_allclose(core[0]['geometry']['size'], PALM_CORE_SIZE_M, atol=1e-9)
        normal = np.asarray(PALM_NORMAL_AXES_BODY[side])
        front = np.max(np.asarray(COATING_SPHERE_CENTERS_BODY_M[side]) @ normal) + COATING_SPHERE_RADIUS_M
        assert abs(front - np.dot(PALM_CENTER_OFFSETS_BODY_M[side], normal)) < 1e-9


def series(trace, time_key, fields, origin):
    times = trace[time_key]
    order = np.unique(times, return_index=True)[1]
    if not np.isfinite(times).all():
        raise ValueError(f'Non-finite timestamps: {time_key}')
    result = dict(t=numbers(times[order]-origin))
    for name, key in fields.items():
        value = trace[key][order]
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Non-finite samples: {key}')
        result[name] = value.tolist() if value.dtype.kind in 'bUS' else numbers(value)
    return result


def episode(path, urdf, check=False):
    meta = json.loads((path / 'meta.json').read_text())
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != meta['posture']['urdf_sha256']:
        raise ValueError(f'{path.name}: URDF checksum differs from the recorded robot')
    with np.load(path / 'trace.npz', allow_pickle=False) as trace:
        obs_times = trace['observation_time_s']
        if 'observation_box_pose' in trace:
            error = np.linalg.norm(trace['observation_position_m'] - trace['observation_box_pose'][:3], axis=1)
            index = int(np.argmin(error))
            if error[index] > 1e-8:
                raise ValueError(f'{path.name}: cannot find the recorded commit observation')
            commit = float(obs_times[index])
        else:
            commit = float(obs_times[-1])
        ct, tp = trace['command_t_host_s'], trace['command_t_plan_s']
        moving = np.flatnonzero((trace['command_phase'] == 'execute') & (tp > 0))[:5]
        origin = float(np.median(ct[moving]-tp[moving])) if len(moving) else commit
        data = dict(name=path.name, decision=meta['decision'], reason=meta['reason'],
                    posture=meta['posture'], origin_host_s=origin, commit=commit-origin,
                    box_dimensions=numbers(2*np.array(meta['catch_settings']['box_half_extents'])),
                    box=series(trace, 'observation_time_s', dict(p='observation_position_m',
                        q='observation_quat_xyzw', valid='observation_valid'), origin),
                    command=series(trace, 'command_t_host_s', dict(p='target_palm_position',
                        q='target_palm_rotation_xyzw', phase='command_phase'), origin), feedback=None,
                    contact=None, flight=None, touch=[], normal=None)
        if 'feedback_t_s' in trace:
            data['feedback'] = series(trace, 'feedback_t_s', dict(left='feedback_joint_left_rad',
                right='feedback_joint_right_rad', p='feedback_palm_position',
                q='feedback_palm_rotation_xyzw'), origin)
            if check:
                joints = parse_joints(urdf)
                max_error = 0.
                for i in range(len(trace['feedback_t_s'])):
                    angles = named_angles(meta['posture']['legwaist_joint_deg'],
                        np.rad2deg(trace['feedback_joint_left_rad'][i]),
                        np.rad2deg(trace['feedback_joint_right_rad'][i]))
                    p, q = palms_from_angles(joints, angles)
                    max_error = max(max_error, float(np.max(abs(p-trace['feedback_palm_position'][i]))))
                    rotation_error = (Rotation.from_quat(q).inv() *
                                      Rotation.from_quat(trace['feedback_palm_rotation_xyzw'][i])).magnitude()
                    assert np.max(rotation_error) < 1e-8, (path.name, i, rotation_error)
                assert max_error < 1e-8, (path.name, max_error)
        if 'contact_time_s' in trace:
            data['contact'] = float(trace['contact_time_s'])
            data['touch'] = numbers(trace['predicted_touch_times_s'])
            data['normal'] = numbers(trace['contact_normals'])
        if 'estimated_box_position_m' in trace:
            flight = BoxFlight(trace['estimated_box_position_m'], trace['estimated_box_velocity_mps'],
                Rotation.from_quat(trace['estimated_box_rotation_xyzw']),
                trace['estimated_box_angular_velocity_radps'], trace['estimated_box_acceleration_mps2'])
            data['flight'] = dict(p=numbers(flight.position_m), v=numbers(flight.velocity_mps),
                a=numbers(flight.acceleration_mps2), q=numbers(flight.rotation.as_quat()),
                w=numbers(flight.angular_velocity_radps))
        streams = [data['box'], data['command']] + ([data['feedback']] if data['feedback'] else [])
        data['span'] = [min(s['t'][0] for s in streams if s['t']),
                        max(s['t'][-1] for s in streams if s['t'])]
        data['window'] = [max(data['span'][0], -.25), min(data['span'][1], 1.2)]
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='?', type=Path, default=DEFAULT_LOGS)
    parser.add_argument('--urdf', type=Path, default=ROOT / 'data/moz1_boxer.urdf')
    parser.add_argument('--meshes', type=Path, default=DEFAULT_MESHES)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/real_robot_replay/index.html')
    parser.add_argument('--self-check', action='store_true', help='Verify every feedback sample against URDF FK')
    args = parser.parse_args()
    logs, urdf = args.logs.expanduser(), args.urdf.expanduser()
    paths = [logs] if (logs / 'trace.npz').is_file() else sorted(logs.glob('attempt_*/'))
    if not paths:
        parser.error(f'No attempt directories found in {logs}')
    payload = dict(robot=robot_model(urdf, args.meshes.expanduser()),
                   palm_offsets=numbers(PALM_CENTER_OFFSETS_BODY_M),
                   palm_normals=numbers(PALM_NORMAL_AXES_BODY),
                   episodes=[episode(path, urdf, args.self_check) for path in paths])
    if args.self_check:
        check_hand_model(payload['robot'])
    encoded = packed(json.dumps(payload, ensure_ascii=False, separators=(',', ':'),
                                allow_nan=False).encode('utf-8'), 'u1')
    template = (ROOT / 'scripts/real_robot_replay.html').read_text()
    vendor = ('/*\n' + (ROOT / 'scripts/replay_vendor/LICENSE.three').read_text() + '\n*/\n' +
              (ROOT / 'scripts/replay_vendor/three-orbit.min.js').read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace('__THREE_VENDOR__', vendor).replace('__REPLAY_DATA__', encoded))
    print(f'{args.output.resolve()} ({args.output.stat().st_size/1024**2:.1f} MiB, {len(paths)} attempts)')
    if args.self_check:
        print('PASS: each runtime hand has 12 coating spheres + one rigid core, with matching palm frames')
        print('PASS: every recorded arm feedback pose matches the recorded URDF FK')


if __name__ == '__main__':
    main()

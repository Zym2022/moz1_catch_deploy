three-orbit.min.js bundles three.js 0.160.1 and OrbitControls, under the MIT
license in LICENSE.three. It is embedded into the exported HTML for offline use.

Built with esbuild 0.24.2 from this entry:

```js
export * from 'three';
export { OrbitControls } from 'three/addons/controls/OrbitControls.js';
```

```sh
esbuild entry.js --bundle --minify --format=iife --global-name=THREE --outfile=three-orbit.min.js
```

Exporting and opening the replay require no npm, CDN, or server.

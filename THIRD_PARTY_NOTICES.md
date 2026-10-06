# Third-party assets

## Microsoft Rocketbox avatars

`src/isharati/app/static/rocketbox_*.vrm` are converted from the Microsoft Rocketbox Avatar Library
(https://github.com/microsoft/Microsoft-Rocketbox): `Female_Adult_06`, `Female_Adult_10`, `Male_Adult_15`,
`Male_Adult_19`, `Male_Adult_21`. Changes: converted from FBX to VRM 1.0 with `scripts/avatars/rocketbox_to_vrm.py`
(textures downscaled to 1024 px, specular maps dropped, Biped bones mapped to the VRM humanoid).

```
MIT License

Copyright (c) 2020 Microsoft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## pixiv VRM sample avatar

`src/isharati/app/static/avatar.vrm` is pixiv Inc.'s `VRM1_Constraint_Twist_Sample` (© 2022 pixiv Inc.), under the VRM
Public License 1.0 (https://vrm.dev/licenses/1.0/): modification, redistribution and commercial and religious use
allowed; credit not required.

## Browser libraries

three.js (MIT) and @pixiv/three-vrm (MIT) are loaded from jsDelivr at run time and are not redistributed here.

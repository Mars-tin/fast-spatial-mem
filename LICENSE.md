# License

This repository contains original code, third-party code, and modifications or derivative works of third-party code. As a result, **this repository is not distributed under a single uniform license**. The license terms applicable to a given file or implementation path depend on that file’s provenance.

Users must comply with the license terms applicable to each file, directory, or implementation path, as described below and in the corresponding license files and file headers.

## 1. Repository structure

This codebase includes two major implementation paths with different upstream provenance.

### 1.1 LRM implementation path

Files in the LRM implementation path include modifications to, or derivative works based on, code originating from some combination of the following upstream sources:

- Gaussian Splatting, originally released under the Gaussian-Splatting License
- PlenOctree, originally released under the BSD 2-Clause License
- downstream derivative work(s) distributed under the Adobe Research License
- DINO and Long-LRM, originally released under the Apache License 2.0
- other third-party components under permissive licenses, including MIT-licensed code
- modifications by Martin Ziqiao Ma

Because this implementation path includes code derived from upstream components distributed under **noncommercial** research licenses, **these files must be treated as subject to the applicable upstream restrictions and notices**, including any noncommercial-use limitations, redistribution conditions, and attribution requirements carried by those upstream licenses.

In particular, the presence of permissively licensed subcomponents does not eliminate or override the restrictions imposed by more restrictive upstream components in the same derivative chain.

See:
- `licenses/LICENSE_GAUSSIAN_SPLATTING.md`
- `licenses/LICENSE_TTTLRM.md`

### 1.2 LVSM implementation path

Files in the LVSM implementation path include modifications to, or derivative works based on, code originating from some combination of the following upstream sources:

- DINO, originally released under the Apache License 2.0
- LaCT and/or other MIT-licensed components
- modifications by Martin Ziqiao Ma

To the extent a file in this implementation path is derived only from Apache-2.0- and MIT-licensed sources, and does not incorporate or derive from code in the more restrictive LRM path, that file may be distributed under Apache License 2.0 together with any required third-party notices.

However, file-level provenance controls. If a file header, subdirectory notice, or third-party notice identifies additional upstream provenance or different applicable terms, those more specific terms govern that file.

See:
- `licenses/LICENSE_DINO`
- `licenses/LICENSE_LACT`

## 2. Original contributions

Unless otherwise stated in a file header, subdirectory notice, or third-party notice, original code authored by Martin Ziqiao Ma that is not a modification of, or derivative work based on, upstream source code is released under the Apache License 2.0.

For clarity, this applies only to genuinely original portions of the repository. It does **not** relicense, weaken, or supersede the terms applicable to any third-party code or upstream-derived files.

## 3. File-level and directory-level control

Where a file header, subdirectory notice, or bundled third-party notice specifies provenance or licensing information for a particular file or directory, that more specific notice controls over this top-level summary.

If a file is a modification of or derivative work based on upstream code, the upstream license terms that apply to that file continue to apply, together with any additional notices required by the corresponding upstream license.

## 4. Redistribution and attribution

Redistributions of this repository, in whole or in part, must preserve:

- applicable upstream license texts,
- copyright notices,
- attribution notices,
- disclaimer notices, and
- any file- or directory-specific provenance statements.

See `NOTICE.md` for additional copyright attribution, provenance information, and third-party component notices.

## 5. No single blanket permission

The inclusion of some files under Apache License 2.0 or other permissive licenses does **not** mean that the repository as a whole may be used under those terms alone.

In particular, no permission is granted to use, redistribute, or commercially exploit files derived from upstream components under the Adobe Research License, the Gaussian-Splatting License, or any other restrictive upstream license except as permitted by those licenses.
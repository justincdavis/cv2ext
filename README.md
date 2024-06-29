# cv2ext

[![](https://img.shields.io/pypi/pyversions/cv2ext.svg)](https://pypi.org/pypi/cv2ext/)
![PyPI](https://img.shields.io/pypi/v/cv2ext.svg?style=plastic)
[![CodeFactor](https://www.codefactor.io/repository/github/justincdavis/cv2ext/badge)](https://www.codefactor.io/repository/github/justincdavis/cv2ext)

![Linux](https://github.com/justincdavis/cv2ext/actions/workflows/unittests-ubuntu.yaml/badge.svg?branch=main)
![Windows](https://github.com/justincdavis/cv2ext/actions/workflows/unittests-windows.yaml/badge.svg?branch=main)
![MacOS](https://github.com/justincdavis/cv2ext/actions/workflows/unittests-macos.yaml/badge.svg?branch=main)

![MyPy](https://github.com/justincdavis/cv2ext/actions/workflows/mypy.yaml/badge.svg?branch=main)
![Ruff](https://github.com/justincdavis/cv2ext/actions/workflows/ruff.yaml/badge.svg?branch=main)
![PyPi Build](https://github.com/justincdavis/cv2ext/actions/workflows/build-check.yaml/badge.svg?branch=main)

A collection of tools for making working with OpenCV in Python easier (and potentially faster).

---

## Documentation

https://cv2tools.readthedocs.io/en/latest/

See Also:

https://pypi.org/project/cv2ext/


## Performance

Compared to the naive solution implemented in Python,
by using the tools in cv2ext you can achieve an 8x speedup
on reading and displaying videos.

![Performance](benchmarks/visual/showplot.png)

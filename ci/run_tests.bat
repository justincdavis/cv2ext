@echo off
setlocal enabledelayedexpansion

python3 -m pytest --log-cli-level=WARNING -rP .\tests\test_same.py || exit /b 1

endlocal

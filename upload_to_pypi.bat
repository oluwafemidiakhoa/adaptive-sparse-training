@echo off
echo ========================================
echo   Uploading to PyPI
echo ========================================
echo.
echo When prompted:
echo   Username: __token__
echo   Password: [paste your ACCOUNT-SCOPED token]
echo.
echo Press any key to start upload...
pause >nul

twine upload dist/*

echo.
echo ========================================
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS! Package uploaded to PyPI
    echo Visit: https://pypi.org/project/adaptive-sparse-training/
) else (
    echo FAILED! Check error message above
)
echo ========================================
pause

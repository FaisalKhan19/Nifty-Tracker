@echo off
setlocal enabledelayedexpansion

REM Path to python
set PYTHON=python

REM CSV file with symbols
set CSV_FILE=ind_nifty100list.csv

REM Python script name
set PY_SCRIPT1=smc_plot.py
set PY_SCRIPT2=choch.py
REM Loop through CSV file, skipping the header
REM Loop through CSV file, skip header, extract 3rd column (Symbol)
for /f "skip=1 tokens=1,2,3 delims=," %%A in (%CSV_FILE%) do (
    set SYMBOL=%%C
    if not "!SYMBOL!"=="" (
        echo Processing !SYMBOL! ...
        %PYTHON% %PY_SCRIPT1% --ticker !SYMBOL!
        %PYTHON% %PY_SCRIPT2% --ticker !SYMBOL!
    )
)

echo All done!

@echo off
cd agi-project
pip install -e .
set PYTHONPATH=%CD%\src
python -m pytest tests\test_agent_logic.py
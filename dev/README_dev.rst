===============
nerblackbox DEV
===============

Unit Testing
============

::

    python dev/check --coverage

output file can be found at `./dev/coverage/index.html`

End-to-end Testing
==================

::

    pytest e2e_tests/e2e_test_api.py
    pytest e2e_tests/e2e_test_cli.py
    pytest e2e_tests/e2e_test_evaluation.py

output files (stdout & stderr) can be found at `./e2e_tests/e2e_test_???_data`


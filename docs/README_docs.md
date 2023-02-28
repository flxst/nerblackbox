CREATE THE DOCUMENTATION
------------------------

1. make sure packages are installed:

    ``pip install -e .[dev]``


2. render the documentation locally

    ``cd docs && mkdocs serve``


3. push the documentation to origin

    ``cd docs && mkdocs gh-deploy -m "vx.y.z" [-b "branch"] [--force]``


4. push the documentation to github (branch `gh-pages`) manually
   
    ``git checkout gh-pages && git push origin gh-pages``

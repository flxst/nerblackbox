
#########################
# CREATE AUTODOC *.rst
#########################
echo
echo '=== CREATE AUTODOC ==='

# needs to be run from docs directory
DIR=./source

sphinx-apidoc -f -o ${DIR} ../nerblackbox ../nerblackbox/modules
cd ${DIR}

sed -e '/Submodule/{N;N;d;}' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst                     # remove Submodule + 1 line
sed -e '/Subpackages/{N;N;N;N;N;N;N;d;}' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst         # remove Submodule + 6 lines
sed -e '/Module contents/{N;N;N;N;N;N;N;d;}' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst     # remove Module contents + 6 lines
sed '/nerblackbox package/i \
  \
  .. _apidocumentation:\
  \
  ' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst                           # replace
sed 's/nerblackbox package/API Documentation/g' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst                           # replace
sed '/nerblackbox.api module/i \
Content: \
\
- :ref:`python_api`\
- :ref:`cli`\
 \
 .. _python_api:\
 \
 ' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst                                               # insert reference
sed 's/nerblackbox.api module/Python API/g' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst      # replace
sed '/nerblackbox.cli module/i \
 .. automodule:: nerblackbox.modules.experiment_results \
    :replace1:\
    :replace2:\
    :replace3:\
 \
 .. automodule:: nerblackbox.modules.experiments_results \
    :replace1:\
    :replace2:\
    :replace3:\
 \
 .. _cli:\
 \
 ' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst                                               # insert reference
sed 's/:replace1:/   :members:/g' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst                # replace
sed 's/:replace2:/   :undoc-members:/g' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst          # replace
sed 's/:replace3:/   :show-inheritance:/g' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst       # replace
sed 's/nerblackbox.cli module/CLI/g' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst             # replace
echo '.. click:: nerblackbox.cli:main
   :prog: nerbb
   :show-nested:' >> nerblackbox.rst

echo "rm ./nerblackbox.tests.rst"
rm nerblackbox.tests.rst

echo "rm ./modules.rst"
rm modules.rst

echo "mv ./nerblackbox.rst ./apidocumentation.rst"
mv nerblackbox.rst apidocumentation.rst
cd ..

#########################
# CREATE HTML
#########################
echo
echo '=== CREATE HTML ==='
make html

#########################
# COPY TO DOCS
#########################
echo
echo '=== COPY TO DOCS ==='
cp -r _build/html/* ../docs

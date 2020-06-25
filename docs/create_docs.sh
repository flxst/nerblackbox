# needs to be run from docs directory
sphinx-apidoc -f -o . ../nerblackbox ../nerblackbox/modules

sed -e '/Submodule/{N;N;d;}' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst                     # remove Submodule + 1 line
sed -e '/Module contents/{N;N;N;N;N;N;N;d;}' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst     # remove Module contents + 6 lines
sed 's/nerblackbox package/API Documentation/g' nerblackbox.rst > temp.rst; mv temp.rst nerblackbox.rst  # replace
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


#########################
# REMOVE PREVIOUS HTML
#########################
echo
echo '=== REMOVE PREVIOUS HTML ==='
rm -r _build

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

pdoc --html flow -o docs --force --config='latex_math=True'
mv docs/flow/* docs/
rm -r docs/flow
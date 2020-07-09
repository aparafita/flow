rm dist/*
python setup.py sdist
twine upload --repository dist/*

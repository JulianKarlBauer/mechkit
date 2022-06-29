# Run tests against installation (recommended)
- Install mechkit in development mode `python setup.py develop`
- Install pytest `pip install pytest`
- Run tests `pytest`
- Uninstall `python setup.py develop --uninstall`

# Run tests against local files
`python -m pytest`  

# Run specific tests
[source](https://stackoverflow.com/a/62804929/8935243)

Example from within ipython
```python
!pytest test/test_notation.py::Test_ExplicitConverter::test_loop_minor_sym --verbose --pdb
```




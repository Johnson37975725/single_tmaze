class Environment:
    def evaluate(self, phenotype):
        assert False, "'evaluate' must be implemented"

if __name__ == '__main__':
    # a test for evaluate
    flag = False
    try:
        Environment().evaluate('phenotype')
    except AssertionError:
        flag = True
    assert flag, "NG in 'environment.py'"

"""Context cleanup for contract fixtures on every supported Python version."""


def enter_context(test, context):
    value = context.__enter__()
    test.addCleanup(context.__exit__, None, None, None)
    return value


def enter_class_context(test_class, context):
    value = context.__enter__()
    test_class.addClassCleanup(context.__exit__, None, None, None)
    return value

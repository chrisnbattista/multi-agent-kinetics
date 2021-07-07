def name(n):
    def wrapper(f):
        f.name = n
        return f
    return wrapper
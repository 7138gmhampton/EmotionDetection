def progress_bar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', print_end='\r'):
    total = len(iterable)
    
    def print_progress_bar(iteration):
        percent = ('{0:.' + str(decimals) + 'f}').format(100*(iteration/float(total)))
        filled_length = int(length*iteration//total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = print_end)
        
    print_progress_bar(0)
    
    for iii, item in enumerate(iterable):
        yield item
        print_progress_bar(iii + 1)
        
    print()
"""Simple progress bar implementation"""
def progress_bar(iterable, decimals=1, length=100, fill='█', print_end='\r', **kwargs):
    """
    Displays progress bar for the iterable loop
    :param prefix: Text to appear before bar
    :param suffix: Text to appear after bar
    """
    total = len(iterable)
    # if 'suffix' in kwargs: suffix = kwargs['suffix']
    # else: suffix = ''
    suffix = kwargs['suffix'] if 'suffix' in kwargs else ''
    prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
    
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
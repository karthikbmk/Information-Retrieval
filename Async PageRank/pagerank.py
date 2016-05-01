""" Assignment 6: PageRank. """
from bs4 import BeautifulSoup
from sortedcontainers import SortedList, SortedSet, SortedDict
from collections import Counter, defaultdict
import glob
import os

def compute_pagerank(urls, inlinks, outlinks, b=.85, iters=20):
    """ Return a dictionary mapping each url to its PageRank.
    The formula is R(u) = (1/N)(1-b) + b * (sum_{w in B_u} R(w) / (|F_w|)

    Initialize all scores to 1.0.

    Params:
      urls.......SortedList of urls (names)
      inlinks....SortedDict mapping url to list of in links (backlinks)
      outlinks...Sorteddict mapping url to list of outlinks
    Returns:
      A SortedDict mapping url to its final PageRank value (float)

    >>> urls = SortedList(['a', 'b', 'c'])
    >>> inlinks = SortedDict({'a': ['c'], 'b': set(['a']), 'c': set(['a', 'b'])})
    >>> outlinks = SortedDict({'a': ['b', 'c'], 'b': set(['c']), 'c': set(['a'])})
    >>> sorted(compute_pagerank(urls, inlinks, outlinks, b=.5, iters=0).items())
    [('a', 1.0), ('b', 1.0), ('c', 1.0)]
    >>> iter1 = compute_pagerank(urls, inlinks, outlinks, b=.5, iters=1)
    >>> iter1['a']  # doctest:+ELLIPSIS
    0.6666...
    >>> iter1['b']  # doctest:+ELLIPSIS
    0.333...
    """
    ###TODO
        
    #Initializations
    N = len(urls)
    page_rank = defaultdict(lambda:1.0)
    for url in urls:
        page_rank[url]
    
    #Iterate to update the ranks
    part1 = (float(1)/(N))*(1-b)    
    for i in range(iters):
        for url in urls:            
            part2 = 0             
            for inlink in inlinks[url]:                                                                          
                part2 += page_rank[inlink]/len(outlinks[inlink])
            page_rank[url] = part1 + (part2*b)
                    
    
    return SortedDict(page_rank)


def get_top_pageranks(inlinks, outlinks, b, n=50, iters=20):
    """
    >>> inlinks = SortedDict({'a': ['c'], 'b': set(['a']), 'c': set(['a', 'b'])})
    >>> outlinks = SortedDict({'a': ['b', 'c'], 'b': set(['c']), 'c': set(['a'])})
    >>> res = get_top_pageranks(inlinks, outlinks, b=.5, n=2, iters=1)
    >>> len(res)
    2
    >>> res[0]  # doctest:+ELLIPSIS
    ('a', 0.6666...
    """
    ###TODO
    urls = SortedList(set(set(inlinks.keys()).union(outlinks.keys())))
    page_ranks = sorted(compute_pagerank(urls, inlinks, outlinks, b, iters).items(), key=lambda x: x[1],reverse=True)
    return page_ranks[0:n]


def read_names(path):
    """ Do not mofify. Returns a SortedSet of names in the data directory. """
    return SortedSet([os.path.basename(n) for n in glob.glob(path + os.sep + '*')])


def get_links(names, html):
    """
    Return a SortedSet of computer scientist names that are linked from this
    html page. The return set is restricted to those people in the provided
    set of names.  The returned list should contain no duplicates.

    Params:
      names....A SortedSet of computer scientist names, one per filename.
      html.....A string representing one html page.
    Returns:
      A SortedSet of names of linked computer scientists on this html page, restricted to
      elements of the set of provided names.

    >>> get_links({'Gerald_Jay_Sussman'},
    ... '''<a href="/wiki/Gerald_Jay_Sussman">xx</a> and <a href="/wiki/Not_Me">xx</a>''')
    SortedSet(['Gerald_Jay_Sussman'], key=None, load=1000)
    """
    ###TODO
    out_linked_names  = []
    
    soup = BeautifulSoup(html, "html.parser")
    hyper_links = soup.findAll("a")
    
    for link in hyper_links:        
        tag_content = link.get("href")              
        if tag_content != None and tag_content.split('/')[-1:][0] in names:            
                out_linked_names.append(tag_content.split('/')[-1:][0])
    
    return SortedSet(out_linked_names)
    

def read_links(path):
    """
    Read the html pages in the data folder. Create and return two SortedDicts:
      inlinks: maps from a name to a SortedSet of names that link to it.
      outlinks: maps from a name to a SortedSet of names that it links to.
    For example:
    inlinks['Ada_Lovelace'] = SortedSet(['Charles_Babbage', 'David_Gelernter'], key=None, load=1000)
    outlinks['Ada_Lovelace'] = SortedSet(['Alan_Turing', 'Charles_Babbage'], key=None, load=1000)

    You should use the read_names and get_links function above.

    Params:
      path...the name of the data directory ('data')
    Returns:
      A (inlinks, outlinks) tuple, as defined above (i.e., two SortedDicts)
    """
    ###TODO
    
    inlinks = defaultdict(lambda:[])
    outlinks = defaultdict(lambda:[])        
    
    names = read_names(path)
    
    #Windows Fix.
    for name in names:
        if name == 'Guy_L._Steele,_Jr':            
            name = 'Guy_L._Steele,_Jr.'
        inlinks[name]
    
    #Windows Fix.
    cpy_names = list(names)
    if 'Guy_L._Steele,_Jr' in cpy_names:
        cpy_names[cpy_names.index('Guy_L._Steele,_Jr')] = 'Guy_L._Steele,_Jr.'
    
    cpy_names = SortedSet(cpy_names)
    
    for name in names:        
        html = open(os.path.join(path,name),encoding="utf-8").read()        

        #Windows Fix.
        if name == 'Guy_L._Steele,_Jr':
            name = 'Guy_L._Steele,_Jr.'
        
        other_names = cpy_names - [name]
        
        outlinks[name] = get_links(other_names, html)
        
        for out_link_name in outlinks[name]:
            inlinks[out_link_name].append(name)
    
    for k in inlinks:
        inlinks[k] = SortedSet(inlinks[k])
        
    return (inlinks,outlinks)

def print_top_pageranks(topn):
    """ Do not modify. Print a list of name/pagerank tuples. """
    print('Top page ranks:\n%s' % ('\n'.join('%s\t%.5f' % (u, v) for u, v in topn)))


def main():
    """ Do not modify. """
    if not os.path.exists('data'):  # download and unzip data        
        from urllib.request import urlretrieve
        import tarfile
        urlretrieve('http://cs.iit.edu/~culotta/cs429/pagerank.tgz', 'pagerank.tgz')
        tar = tarfile.open('pagerank.tgz')
        tar.extractall()
        tar.close()

    inlinks, outlinks = read_links('data')
    print('read %d people with a total of %d inlinks' % (len(inlinks), sum(len(v) for v in inlinks.values())))
    print('read %d people with a total of %d outlinks' % (len(outlinks), sum(len(v) for v in outlinks.values())))
    topn = get_top_pageranks(inlinks, outlinks, b=.8, n=20, iters=10)
    print_top_pageranks(topn)


if __name__ == '__main__':
    main()

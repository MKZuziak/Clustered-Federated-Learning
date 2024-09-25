from itertools import islice

# def chunker(seq: iter, size: int):
#     """Helper function for splitting an iterable into a number
#     of chunks each of size n. If the iterable can not be splitted
#     into an equal number of chunks of size n, chunker will return the
#     left-overs as the last chunk.
        
#     Parameters
#     ----------
#     sqe: iter
#         An iterable object that needs to be splitted into n chunks.
#     size: int
#         A size of individual chunk
#     Returns
#     -------
#     Generator
#         A generator object that can be iterated to obtain chunks of the original iterable.
#     """
#     return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def chunker_dict(seq: dict, size:int):
   it = iter(seq)
   for i in range(0, len(seq), size):
      yield {k:seq[k] for k in islice(it, size)}
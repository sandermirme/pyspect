from collections import namedtuple

from .util import parse_spectops_time

Comment = namedtuple("Comment", ["begin_time", "end_time", "opmode", "text"])


def read_comments(sourcefile):
    res = []
    for line in sourcefile:
        if not line:
            continue
        if line[0] == "#":
            continue
        res.append(Comment(*(line.rstrip("\n\r").split("\t"))))

    return comments_sorted(res)


def comment_sort_key(comment):
    return parse_spectops_time(comment[0])


def comments_sorted(comments):
    return sorted(comments, key=comment_sort_key)

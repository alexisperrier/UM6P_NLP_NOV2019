'''
Regex
'''

# Rabbits in Alice in Wonderland
import ntlk
import re

alice = nltk.corpus.gutenberg.open('carroll-alice.txt').read()

'''
Search specific word
'''
match = re.search("White Rabbit", alice)

match = re.search("rabbit", alice)
match.start();
match.end()

match = re.search("rabbot", alice)

# findall

match = re.findall("Rabbit", alice)

# what about lower case

match = re.search("[R|r]abbit", alice)
# or
match = re.search("rabbit", alice, , flags=re.IGNORECASE)

# Iterations
for m in re.finditer("[R|r]abbit", alice):
    print(alice[m.start(): m.end() ])

for m in re.finditer("[R|r]abbits?-?", alice):
    print(alice[m.start(): m.end() ])


'''
Part II: tweets abeilles
- trouver les hashtags
- trouver les urls
- trouver les @
'''

import pandas as pd
df = pd.read_csv('abeilles_tweets.csv')
df.sample(10).main.values

tweets = '\n'.join(df.main.values)

# 1. extraire tous les hashtags
# for m in re.finditer("#.+\s", tweets[0:1000]):
#     print(tweets[m.start(): m.end() ])

for m in re.finditer(r"#\w+", tweets[0:1000]):
    print(m.group())

# handles (@N_hulot, @truc, ...) les plus fréquents
handles = re.findall(r"@\w+", tweets)
from collections import Counter
Counter(handles).most_common(10)


# remplacer les hashtags
nohtags = re.sub(regex, '---', tweets)




'''
US phone numbers
'''
rgx = r"\d\d\d-\d\d\d-\d\d\d\d"
rgx = r"(\d{3}-)?\d{3}-\d{4}"


'''
Substituting text with regular expressions
regex.sub(s1, s2)
'''


>>> import re
>>> string = "If the the problem is textual, use the the re module"
>>> pattern = r"the the"
>>> regexp = re.compile(pattern)
>>> regexp.sub("the", string)
'If the problem is textual, use the re module'




# ------

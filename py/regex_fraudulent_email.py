'''
regex_fraudulent_email.txt
based on https://www.kaggle.com/rtatman/fraudulent-email-corpus
and https://www.dataquest.io/blog/regular-expressions-data-scientists/

'''


# read the file
fh = open("../data/fradulent_emails.txt", "r",encoding='windows-1252').read()

'''
We look for the lines that start with "From:"
'''

import re

from_lines = []
for line in re.findall("From:.*", fh):
    from_lines.append(line)

print(from_lines[:10])

'''
The dot: . indicates any character except \n
the star: * allows multiple occurences of the preceding element: here the . (any character)
what happens if we just search for "From:" then we only get "From:" in return not the whole line
'''


'''
Now in each of the line that was found,
Extract the emails
'''

from_lines = []
emails = []
for line in re.findall("From:.*", fh):
    from_lines.append(line)
    for email in re.findall("<.*>",line):
        emails.append(email)

'''
pattern= any characters between <>
'''

'''
Look for email by defining a real email pattern
'''
regex = "\wS*@.*\w"
from_lines = []
emails = []
for line in re.findall("From:.*", fh):
    from_lines.append(line)
    for email in re.findall(regex, line):
        emails.append(email)

Better Language Modelsand Their Implications

# --------------

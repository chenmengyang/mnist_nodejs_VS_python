import os
import re

# empty line pattern
emptyPat = re.compile('^$')

#  check how many lines of code under the path
def checkSourceCode(fileName)-> int: 
    # return how many lines of code in the file, ignore comments
    commentSymbol = '#' if fileName.endswith('py') else '//'
    outputPattern = 'print\s?\(.*\)' if fileName.endswith('py') else 'console.log\(.*\)'
    n = 0
    with open(fileName, 'r') as f:
        for l in f.readlines():
            # first remove empty spaces at the head of line
            l = re.sub(outputPattern,'',l)
            l = re.sub('^\s+','',l)
            if (emptyPat.match(l) == None) and (l.startswith(commentSymbol) is False):
                n+=1
    return n

nodejsFiles = (f for f in os.listdir('./nodejs') if f.endswith('.js'))
codeLines = (checkSourceCode(os.path.join('./nodejs', jsFile)) for jsFile in nodejsFiles)
print('There are {} lines of nodejs codes'.format(sum(codeLines)))
# spaCy学习笔记

[spaCy](https://spacy.io/)是一个Python工业级的nlp处理库， spaCy里大量使用了 [Cython](http://cython.org/) 来提高相关模块的性能，这个区别于学术性质更浓的Python NLTK，因此具有了业界应用的实际价值。

### 安装

安装spaCy

```python
pip install -U spacy
```

安装module

```python
python -m spacy download en_core_web_sm
```

或者可以分别下载相关的模型和用glove训练好的词向量数据

```python
# 这个过程下载英文tokenizer，词性标注，句法分析，命名实体识别相关的模型
python -m spacy.en.download parser

# 这个过程下载glove训练好的词向量数据
python -m spacy.en.download glove
```

### 引入和加载

```python
import spacy
nlp = spacy.load("en_core_web_sm")
```

### tokenizer

```python
test_doc = nlp(u"it's word tokenize test for spacy")

print(test_doc)
it's word tokenize test for spacy

for token in test_doc:
    print(token)
...:
it
's
word
tokenize
test
for
spacy
```

### **Lemmatize**

```python
test_doc = nlp(u"you are best. it is lemmatize test for spacy. I love these books")

for token in test_doc:
    print(token, token.lemma_, token.lemma)
...:
(you, u'you', 472)
(are, u'be', 488)
(best, u'good', 556)
(., u'.', 419)
(it, u'it', 473)
(is, u'be', 488)
(lemmatize, u'lemmatize', 1510296)
(test, u'test', 1351)
(for, u'for', 480)
(spacy, u'spacy', 173783)
(., u'.', 419)
(I, u'i', 570)
(love, u'love', 644)
(these, u'these', 642)
(books, u'book', 1011)
```

###  **POS Tagging\(识别单词的词性，**[**具体的表示**](https://spacy.io/api/annotation#pos-tagging)**\):**

```python
# Pos为粗度的词性标注，更为细粒度的词性标注为Tag
for token in test_doc:
    print(token, token.pos_, token.pos)
....:
(you, u'PRON', 92)
(are, u'VERB', 97)
(best, u'ADJ', 82)
(., u'PUNCT', 94)
(it, u'PRON', 92)
(is, u'VERB', 97)
(lemmatize, u'ADJ', 82)
(test, u'NOUN', 89)
(for, u'ADP', 83)
(spacy, u'NOUN', 89)
(., u'PUNCT', 94)
(I, u'PRON', 92)
(love, u'VERB', 97)
(these, u'DET', 87)
(books, u'NOUN', 89)
```

### Dependency\(句法分析之后的单词在句子中的成分，[具体表示](https://spacy.io/api/annotation#dependency-parsing)\):

```python
for token in test_doc:
    print(token, token.dep_, token.dep)
....:
(you, u'nsubj',  429)
(are, u'ROOT', 8206900633647566924)
(best, u'acomp', 398)
(., u'punct', 445)
(it, u'nsubj', 429)
(is, u'ROOT', 8206900633647566924)
(lemmatize, u'xcomp', 450)
(test, u'dobj', 416)
(for, u'prep', 443)
(spacy, u'pobj', 439)
(., u'punct', 445)
(I, u'nsubj', 429)
(love, u'ROOT', 8206900633647566924)
(these, u'det', 415)
(books, u'dobj', 416)
```

### 命名实体识别：

```python
In [11]: test_doc = nlp(u"Rami Eid is studying at Stony Brook University in New York")

In [12]: for ent in test_doc.ents:
print(ent, ent.label_, ent.label)
....:
(Rami Eid, u'PERSON', 346)
(Stony Brook University, u'ORG', 349)
(New York, u'GPE', 350)
```

### noun\_chunks:

```python
In [13]: test_doc = nlp(u'Natural language processing (NLP) deals with the application of computational models to text or speech data. Application areas within NLP include automatic (machine) translation between languages; dialogue systems, which allow a human to interact with a machine using natural language; and information extraction, where the goal is to transform unstructured text into structured (database) representations that can be searched and browsed in flexible ways. NLP technologies are having a dramatic impact on the way people interact with computers, on the way people interact with each other through the use of language, and on the way people access the vast amount of linguistic data now in electronic form. From a scientific viewpoint, NLP involves fundamental questions of how to structure formal models (for example statistical models) of natural language phenomena, and of how to design algorithms that implement these models.')

In [14]: for np in test_doc.noun_chunks:
print(np)
....:
Natural language processing
Natural language processing (NLP) deals
the application
computational models
text
speech
data
Application areas
NLP
automatic (machine) translation
languages
dialogue systems
a human
a machine
natural language
information extraction
the goal
unstructured text
structured (database) representations
flexible ways
NLP technologies
a dramatic impact
the way
people
computers
the way
people
the use
language
the way
people
the vast amount
linguistic data
electronic form
a scientific viewpoint
NLP
fundamental questions
formal models
example
natural language phenomena
algorithms
these models
```

### 计算两个单词相似度：

```python
In [13]: test_doc = nlp(u'Natural language processing (NLP) deals with the application of computational models to text or speech data. Application areas within NLP include automatic (machine) translation between languages; dialogue systems, which allow a human to interact with a machine using natural language; and information extraction, where the goal is to transform unstructured text into structured (database) representations that can be searched and browsed in flexible ways. NLP technologies are having a dramatic impact on the way people interact with computers, on the way people interact with each other through the use of language, and on the way people access the vast amount of linguistic data now in electronic form. From a scientific viewpoint, NLP involves fundamental questions of how to structure formal models (for example statistical models) of natural language phenomena, and of how to design algorithms that implement these models.')

In [14]: for np in test_doc.noun_chunks:
print(np)
....:
Natural language processing
Natural language processing (NLP) deals
the application
computational models
text
speech
data
Application areas
NLP
automatic (machine) translation
languages
dialogue systems
a human
a machine
natural language
information extraction
the goal
unstructured text
structured (database) representations
flexible ways
NLP technologies
a dramatic impact
the way
people
computers
the way
people
the use
language
the way
people
the vast amount
linguistic data
electronic form
a scientific viewpoint
NLP
fundamental questions
formal models
example
natural language phenomena
algorithms
these models
```

### [Retokenizer](https://spacy.io/api/doc#retokenizer.merge):

```python
def tag_fixer_after_parser(doc):
    for token in doc:
        if token.dep_ == 'conj' and token.pos_ == 'NOUN' and token.head.pos_ == 'VERB':
            attrs = {"LEMMA": token.lemma_, "TAG": token.head.tag_, "POS": "VERB"}
            with doc.retokenize() as retokenizer:
                retokenizer.merge(doc[token.i:token.i+1], attrs=attrs)
    return doc
```

### [Pipeline and Extension](https://explosion.ai/blog/spacy-v2-pipelines-extensions):

* extension

```python
import spacy
from spacy.tokens import Doc

Doc.set_attribute('is_greeting', default=False)

nlp = spacy.load('en')
doc = nlp(u'hello world')
doc._.is_greeting = True
```

* pipeline

```python
def tag_fixer_after_parser(doc):
    for token in doc:
        if token.dep_ == 'conj' and token.pos_ == 'NOUN' and token.head.pos_ == 'VERB':
            attrs = {"LEMMA": token.lemma_, "TAG": token.head.tag_, "POS": "VERB"}
            with doc.retokenize() as retokenizer:
                retokenizer.merge(doc[token.i:token.i+1], attrs=attrs)
    return doc
nlp = spacy.load('en')
nlp.add_pipe(tag_fixer_after_parser, name='fp', after='parser')
doc = nlp(u"This is a sentence.")
```

### 分句抽取：

首先获取句子的root，如果句子的root是动词，且包含主语\(nsubj,nsubpass\)，那么就是个完整的句子，就把句子输入到分句抽取中。

* 主要是处理定语从句，e.g. an object that does not support the Serializable interface may be encountered， 定语从句有一个`dep_:relcl`，可以进行判断，如果其头结点是名词，那么就是一个定语从句，进行分词截取处理。
* 连词双动词句子, e.g. This implementation provides all of the optional map operations, and permits null values and the null key. 通过判断是否存在有动词的`dep_`为`conj`且该动词的head为predicate（root），那么就说明存在并列的两个动词，然后进行处理。
* 状语从句，e.g. This class is roughly equivalent to Vector, except that this class writes fast，`dep_`为`mark`，`tag_`为`in`，就是状语从句。

### 抽取模板：

```javascript
AE = API ELEMENT
1. AE belong to NN

    for category extraction, but the relation between NNs is 'part of'. e.g. the method belong to the class.

2. AE have NN

    for category extraction, but the relation between NNs is 'part of'. e.g. The class has the method.

3. AE is like NN

    for functionality extraction, it is just like the (same as template), but 'like' in this template is a prep. e.g. Stringbuffer is like StringBuilder.

4. AE represent NN

    for category extraction, relation between NNs is 'is a'. e.g. stringbuffer represent java.lang.StringBuffer

5. AE be VBN to do

    for functionality extraction, 'VBN' cannot represent the real functionality, so filter it.  e.g. the class is designed to read file.

6. AE be VBN for doing

    for functionality extraction, 'VBN' cannot represent the functionality, so filter it. e.g. the class is used for reading file.

7. AE provide for doing

    for functionality extraction, 'provide' cannot represent the functionality, so filter it. e.g. the class provides for reading files.

8. AE be to do

    for functionality extraction, filter 'be to' and get the verb behind 'to' to extract the real functionality.  e.g. The offer_method is to just offer to the queue

9. AE be (adj) for doing

    for functionality extraction, filter 'be (adj) for' and get the verb behind 'for' to extract the real functionality. e.g. FileInputStream is for reading streams of bytes.

10. AE be RBR (than AE)

    for comparison charateristic extraction, e.g. the class is safer than another

11. AE be more/less adj (than AE)

    for comparison charateristic extraction, e.g. the class is more thread-safe than another

12. AE be VBN by NN

    for functionality extraction, e.g. the value is modified by class B.

13. AE be ADJ* NN

    for category extraction(NP1, NP2) and characteristic extractoin(ADJ), e.g. StringBuilder is a mutable sequence of characters.

14. AE be adj+

    for characteristic extraction, e.g. Instances of StringBufferare thread-safe and mutable.

15. AE be same/equivalent/... prep NN

    for functionality extraction, e.g. This class is roughly equivalent to Vector

16. AE be similar/... prep NN

	for functionality extraction, e.g. The Callable interface is similar to Runnable

17. AE be different/... prep NN

	for functionality extraction, e.g. Stringbuffer is different from StringBuilder.

18. AE MD be VBN

    for characteristic extraction, e.g. The class can be modified.

19. AE VERB adv(except only/always/properly/rather/thus/not/already/...)

    for functionality and characterstic extraction, e.g. Filereader reads file efficiently.

20. AE VERB RBR (than NN)

	for comparison functionality extraction, e.g. The class writes file faster than StringBuilder.

21. AE VERB more/less adv (than NN)

	for comparison functionality extraction, e.g. class A write files faster than class B.

22. AE allow/permit/... NN

    for characteristic extraction, e.g. This implementation permits null values and the null key.

23. AE guarantee/... NN (AE makes no guarantee ...)

	for characteristic extraction, e.g. it does not guarantee the order of the map.

24. AE prohibit/... NN

	for characteristic extraction, e.g. some implementations prohibit null elements

25. AE be adj(whose dep_ is ROOT, but it is an adj.)

    for characterstic extraction. Because the root of sentence satisfy this template is adj, it will be classified into (NN ... VBN) by mistake not (NN be adj), so we create the new template.   e.g. The methods are synchronized

```


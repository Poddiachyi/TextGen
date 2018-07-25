# TextGen

A project created while being a student at https://datarootlabs.com/

See textgen.pdf for a presentation.

An idea was to generate meaningful unique texts by taking one thematic text as a basis.

## Examples:

Taking some sentences from given thematic text and running them through.

| Input  | Output |
| ------------- | ------------- |
| It was not like the boss to make them.  | It was not like our customer support to answer all your questions 24/7.  |
| I sat and read that book for four hours.  | I sat and fished my own traps for four hours.  |
| In ten minutes the doctor came briskly out.  | In ten minutes I fished briskly out.  |

## Solution

Real text generation is a quite complex task. After a few attempts (including similar to https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/) it was decided to simplify it a little bit.

We take a thematic text as a basis. The task is to generate lots of texts based on it. We use IBM Watson to get SOA (subject, object, action) from each sentence in the thematic text and also apply Watson to any other text which will be used for generation (we used a couple of books). Then for each sentence in the thematic text we look for similar sentence in the text for generation. When found, we take SOA from the text for generation and place it in the thematic text.

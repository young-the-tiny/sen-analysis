# Thư viện xử lý chuỗi
import re
import string
# Thư viện xử lý ngôn ngữ tự nhiên
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np # Thư viện tính toán đại số tuyến tính
from contractions import CONTRACTION_MAP
import re
from unidecode import unidecode
from collections import Counter
"""
Input: Text with emojis
Output: Text without emojis
"""
def remove_emoji(text):
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U0001F1F2"
                        u"\U0001F1F4"
                        u"\U0001F620"
                        u"\u200d"
                        u"\u2640-\u2642"
                        u"\u2600-\u2B55"
                        u"\u23cf"
                        u"\u23e9"
                        u"\u231a"
                        u"\ufe0f"  # dingbats
                        u"\u3030"
                        u"\U00002500-\U00002BEF"  # Chinese char
                        u"\U00010000-\U0010ffff"
                        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    """
    Source: https://www.kdnuggets.com/2018/08/practitioners-guide-processing-understanding-text-2.html
    Input: Text with contractions
    Output: Text without contractions
    """
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def preprocess_review(review):
    """
    Return a cleaned text (string).
    
    Input
    review: A review text string.
    -----------------------------------------
    Output
    preprocessed_review: A string without digits, 
    emojis and expanded contractions.
    """
    preproc_review = review.lower() # Lowecase
    
    # Replace happy and sad emoticons by "happy" and "sad" words.
    SAD_FACE = [':(', ':c']
    HAPPY_FACE = [':)', ':D']
    
    for face in SAD_FACE:
        if face in preproc_review:
                preproc_review = preproc_review.replace(face, 'sad')
                
    for face in HAPPY_FACE:
        if face in preproc_review:
                preproc_review = preproc_review.replace(face, 'happy')
        
    # Replace ratings by words
    for match in re.finditer(r'([0-9][0-9]?(\.[0-9])?|100?)\/(100?)', preproc_review):
        numerator =  match.group(1)
        denominator = match.group(3)
        rating = float(numerator) / float(denominator)
        repl_str = f'{numerator}/{denominator}'
        if rating < 0.5:
            preproc_review = preproc_review.replace(repl_str, 'terrible')
        elif rating < 0.6:
            preproc_review = preproc_review.replace(repl_str, 'bad')
        elif rating < 0.8:
            preproc_review = preproc_review.replace(repl_str, 'okay')
        elif rating < 1:
            preproc_review = preproc_review.replace(repl_str, 'good')
        else:
            preproc_review = preproc_review.replace(repl_str, 'excellent')
    # Bỏ các ký tự không phải chữ cái
    preproc_review = preproc_review.translate(str.maketrans('', '', '0123456789'))
    # Bỏ đi các emoji
    preproc_review = remove_emoji(preproc_review)
    # Chuyển các từ loại unicode thành ascii, eg.'İ,Ö,Ü,Ş,Ç,Ğ' -> 'I,O,U,S,C,G'
    preproc_review = unidecode(preproc_review)
    # Chuyển các từ viết tắt thành đầy đủ, eg. "I'll" -> "I will"
    preproc_review = expand_contractions(preproc_review)
    # Bỏ dấu câu
    preproc_review = preproc_review.translate(str.maketrans('', '', string.punctuation)) # string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return preproc_review

def stem_tokenizer(review):
    """
    Trả về một danh sách các từ đã được tách từ một review.
    
    Input
    review: Một chuỗi trong bài đánh.
    --------------------------------------------------------------
    Output
    A list of strings.
    """
    # Tokenization
    review_tokens = word_tokenize(review)
        
    ps = PorterStemmer()
    
    # Stemming
    return [ps.stem(word) for word in review_tokens]
def tokenizer(review):
    """
    Return a collection of tokens extracted from a review.
    
    Input
    review: A review text string.
    --------------------------------------------------------------
    Output
    Một danh sách các chuỗi.
    """
    # Tokenization
    review_tokens = word_tokenize(review)
    
    return review_tokens   
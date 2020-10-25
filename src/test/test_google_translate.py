from src.translation.google_token_generator import generateToken
from src.translation.google_translator import GoogleTranslator, ClientType

if __name__ == '__main__':
    assert (generateToken('Hi There') == str(330723.211101))
    translator = GoogleTranslator(clientType=ClientType.siteGT)
    translation = translator.translate('我爱你！哈哈哈哈')
    print(translation.translated)

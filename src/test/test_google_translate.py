from src.translation.google_token_generator import generate_token
from src.translation.google_translator import GoogleTranslator, ClientType

if __name__ == '__main__':
    assert (generate_token('Hi There') == str(330723.211101))
    translator = GoogleTranslator(client_type=ClientType.siteGT)
    translation = translator.translate('我爱你！哈哈哈哈')
    print(translation.translated)

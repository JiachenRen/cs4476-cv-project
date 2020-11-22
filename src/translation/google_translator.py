from enum import Enum
from urllib import parse
from urllib.request import Request, urlopen
from http.client import HTTPResponse
from json import loads

from src.translation.google_token_generator import generate_token


class ClientType(Enum):
    siteGT = 't'
    extensionGT = 'gtx'


class Translation:
    def __init__(self, source, translated, src_lang, dest_lang):
        self.source = source
        self.translated: str = translated
        self.src_lang = src_lang
        self.dest_lang = dest_lang


class GoogleTranslator:
    """
    Google Translation API wrapper using google translate site's hidden API, tokens are generated on the fly.
    Ported from dart https://github.com/gabrielpacheco23/google-translator/blob/master/lib/src/google_translator.dart
    """
    languages = {
        'auto': 'Automatic',
        'af': 'Afrikaans',
        'sq': 'Albanian',
        'am': 'Amharic',
        'ar': 'Arabic',
        'hy': 'Armenian',
        'az': 'Azerbaijani',
        'eu': 'Basque',
        'be': 'Belarusian',
        'bn': 'Bengali',
        'bs': 'Bosnian',
        'bg': 'Bulgarian',
        'ca': 'Catalan',
        'ceb': 'Cebuano',
        'ny': 'Chichewa',
        'zh-cn': 'Chinese Simplified',
        'zh-tw': 'Chinese Traditional',
        'co': 'Corsican',
        'hr': 'Croatian',
        'cs': 'Czech',
        'da': 'Danish',
        'nl': 'Dutch',
        'en': 'English',
        'eo': 'Esperanto',
        'et': 'Estonian',
        'tl': 'Filipino',
        'fi': 'Finnish',
        'fr': 'French',
        'fy': 'Frisian',
        'gl': 'Galician',
        'ka': 'Georgian',
        'de': 'German',
        'el': 'Greek',
        'gu': 'Gujarati',
        'ht': 'Haitian Creole',
        'ha': 'Hausa',
        'haw': 'Hawaiian',
        'iw': 'Hebrew',
        'hi': 'Hindi',
        'hmn': 'Hmong',
        'hu': 'Hungarian',
        'is': 'Icelandic',
        'ig': 'Igbo',
        'id': 'Indonesian',
        'ga': 'Irish',
        'it': 'Italian',
        'ja': 'Japanese',
        'jw': 'Javanese',
        'kn': 'Kannada',
        'kk': 'Kazakh',
        'km': 'Khmer',
        'ko': 'Korean',
        'ku': 'Kurdish (Kurmanji)',
        'ky': 'Kyrgyz',
        'lo': 'Lao',
        'la': 'Latin',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'lb': 'Luxembourgish',
        'mk': 'Macedonian',
        'mg': 'Malagasy',
        'ms': 'Malay',
        'ml': 'Malayalam',
        'mt': 'Maltese',
        'mi': 'Maori',
        'mr': 'Marathi',
        'mn': 'Mongolian',
        'my': 'Myanmar (Burmese)',
        'ne': 'Nepali',
        'no': 'Norwegian',
        'ps': 'Pashto',
        'fa': 'Persian',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'pa': 'Punjabi',
        'ro': 'Romanian',
        'ru': 'Russian',
        'sm': 'Samoan',
        'gd': 'Scots Gaelic',
        'sr': 'Serbian',
        'st': 'Sesotho',
        'sn': 'Shona',
        'sd': 'Sindhi',
        'si': 'Sinhala',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'so': 'Somali',
        'es': 'Spanish',
        'su': 'Sundanese',
        'sw': 'Swahili',
        'sv': 'Swedish',
        'tg': 'Tajik',
        'ta': 'Tamil',
        'te': 'Telugu',
        'th': 'Thai',
        'tr': 'Turkish',
        'uk': 'Ukrainian',
        'ur': 'Urdu',
        'uz': 'Uzbek',
        'vi': 'Vietnamese',
        'cy': 'Welsh',
        'xh': 'Xhosa',
        'yi': 'Yiddish',
        'yo': 'Yoruba',
        'zu': 'Zulu'
    }

    def __init__(self, client_type: ClientType = ClientType.siteGT):
        self._baseUrl = 'translate.googleapis.com'
        self._path = '/translate_a/single'
        self._client_type = client_type

    def translate(self, src_text: str, src_lang='auto', dest_lang='en'):
        parameters = {
            'client': self._client_type.name,
            'sl': src_lang,
            'tl': dest_lang,
            'hl': dest_lang,
            'dt': 't',
            'ie': 'UTF-8',
            'oe': 'UTF-8',
            'otf': '1',
            'ssel': '0',
            'tsel': '0',
            'kc': '7',
            'tk': generate_token(src_text),
            'q': src_text
        }

        url = f'https://{self._baseUrl}{self._path}?{parse.urlencode(parameters)}'
        req = Request(url)
        with urlopen(req) as response:
            response: HTTPResponse = response
            if response.status != 200:
                raise Exception(f'status code {response.status}')
            data = response.read()
            json = loads(data)
            translated = ''
            for i in range(len(json[0])):
                translated += json[0][i][0]
            if src_lang == 'auto' and src_lang != dest_lang:
                if json[2] is not None:
                    src_lang = json[2]
                if src_lang == dest_lang:
                    src_lang = 'auto'
            return Translation(src_text, translated, src_lang, dest_lang)

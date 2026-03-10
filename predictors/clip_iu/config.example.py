# -*- coding:utf-8 -*-
# ===================================================================
# config.example.py  —  複製此檔並命名為 config.py，填入真實 API key
# cp config.example.py config.py
# ===================================================================
from pathlib import Path
from argparse import Namespace, ArgumentParser


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--lang', type=str, default='en', choices=['en', 'zh'])

    parser.add_argument('--max_turn', type=int, default=4, help='-1 means unlimited')

    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temparature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--frequency_penalty', type=float, default=0.1)
    parser.add_argument('--presence_penalty', type=float, default=0.1)

    # ⚠️  請在 config.py 中填入你的真實 OpenAI API Key，不要提交到 git
    parser.add_argument('--openai_key', type=str, default='sk-REPLACE_WITH_YOUR_OPENAI_API_KEY')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')

    parser.add_argument('--max_token_len', type=int, default=80)
    parser.add_argument('--max_utter_len', type=int, default=160)
    parser.add_argument('--max_bucket_len', type=int, default=250)
    parser.add_argument('--n_gram', type=int, default=4, choices=[3, 4, 5])
    parser.add_argument('--token_hidden_channel_dim', type=int, default=64, choices=[32, 64, 128])
    parser.add_argument('--utter_node_feature_dim', type=int, default=256, choices=[64, 128, 256])
    parser.add_argument('--num_utter_window', type=int, default=3, choices=[3, 4, 5])
    parser.add_argument('--utter_hidden_channel_dim', type=int, default=128, choices=[64, 128, 256])
    parser.add_argument('--emotion_class', type=list, default=['valences', 'arousals', 'dominances'])
    parser.add_argument('--ea_head', type=int, default=8, help='The hidden dimension of External Attention.', choices=[4, 8, 16])
    parser.add_argument('--hidden_dim', type=int, default=128, choices=[128, 256])
    parser.add_argument('--dropout_rate', type=float, default=0.5)

    parser.add_argument('--storage_dir', type=Path, default='./storage')
    parser.add_argument('--vad_trained_path', type=Path, default='./emotion/roberta_emotion_vad_detection.ckpt')

    args = parser.parse_args()

    return args

from slackclient import SlackClient
import os

def s_print(text, channel):
    if channel:
        slack_token = os.environ["SLACK_API_TOKEN"]
        sc = SlackClient(slack_token)
        sc.api_call(
          "chat.postMessage",
          channel='#'+channel,
          text=text
        )
    print(text)
    
if __name__ == '__main__':
    s_print('test', 'output')

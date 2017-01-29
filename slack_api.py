from slackclient import SlackClient
import os

slack_token = os.environ["SLACK_API_TOKEN"]
sc = SlackClient(slack_token)

def send(t):
    sc.api_call(
      "chat.postMessage",
      channel="#output",
      text=t
    )

if __name__ == '__main__':
    send('test')

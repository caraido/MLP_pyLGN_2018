from twilio.rest import Client

account_sid = 'AC54ddb2edef6333de4a9852e448d334ec'
auth_token = '5e0119419b64214fe5a0afbd39e9a3b1'
client = Client(account_sid, auth_token)

message = client.messages.create(
    from_='+17575449618',
    to='+8616621022480',
    body="your simulation is finished!"
)

print(message.sid)
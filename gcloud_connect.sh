gcloud compute os-login ssh-keys add \
  --key-file=/Users/tailia.malloy/.ssh/id_ed25519.pub \
  --project=testgcp-481922 \
  --ttl=1000000000000

(venv) tailia.malloy@UNIJR712W3CKF energy-coding % gcloud compute os-login ssh-keys add \
  --key-file=/Users/tailia.malloy/.ssh/id_ed25519.pub \
  --project=testgcp-481922 \
  --ttl=1000000000000
loginProfile:
  name: '117895056650833064812'
  posixAccounts:
  - accountId: testgcp-481922
    gid: '937979084'
    homeDirectory: /home/tailiamalloy_gmail_com
    name: users/tailiamalloy@gmail.com/projects/testgcp-481922
    operatingSystemType: LINUX
    primary: true
    uid: '937979084'
    username: tailiamalloy_gmail_com
  sshPublicKeys:
    86dc130b1b2f40c3192bbf04678df6f7f0d9079b45cc1217c4f33d29c473cbaa:
      expirationTimeUsec: '1001771343253502464'
      fingerprint: 86dc130b1b2f40c3192bbf04678df6f7f0d9079b45cc1217c4f33d29c473cbaa
      key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ6kuwwxq97KgmqJ5JoXa9nbOMkSjq1nUZISXxV2HfwJ
        tailiamalloy@gmail.com
      name: users/tailiamalloy@gmail.com/sshPublicKeys/86dc130b1b2f40c3192bbf04678df6f7f0d9079b45cc1217c4f33d29c473cbaa


ssh -i /Users/tailia.malloy/.ssh/id_ed25519 tailiamalloy@gmail.com@34.28.128.97


/tailia.malloy/.ssh/id_ed25519.pub
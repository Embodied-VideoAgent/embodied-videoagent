import socket
import pickle


def InternVL2_VQA(question, image_path):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect("tmp/vqa.sock")
    content_file = 'tmp/content.pkl'
    content = dict()
    content['image_path'] = image_path
    content['question'] = question
    with open(content_file, 'wb') as f:
        pickle.dump(content, f)
    client.send(b'sent')
    res = client.recv(1024).decode('utf-8')
    with open(content_file, 'rb') as f:
        ans = pickle.load(f)
    client.send(b'finish')
    return ans

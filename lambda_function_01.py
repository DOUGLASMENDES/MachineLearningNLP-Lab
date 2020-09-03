import json
import boto3
import random


def lambda_handler(event, context):
    print("RECEBENDO: " + json.dumps(event))
    n = random.randint(1,3000)
    response = inserir_item_db("lambda_01", str(n), "Nothing happens at all.", 0)
    valorAClassificar = event["ValorAClassificar"]
    retornoMLTipo, retornoMLProbab = classificar(valorAClassificar)
    return {
        'statusCode': 200,
        'retorno-classificador-binario': json.dumps(retornoMLTipo),
        'retorno-classificador-probab': json.dumps(retornoMLProbab),
        'body': json.dumps(response)
    }


def inserir_item_db(title, id, plot, rating, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb') 

    table = dynamodb.Table('MyTestTable')
    response = table.put_item(
       Item = {
            'id': id,
            'descricao': title
        }
    )
    return response

## event["ValorAClassificar"]
#'debug': json.dumps(event),

def classificar(valor):
    #todo: classifica com modelo em sklearn
    if valor.isnumeric():
        return ("A", 0.7)
    else:
        return ("B", 0.3)



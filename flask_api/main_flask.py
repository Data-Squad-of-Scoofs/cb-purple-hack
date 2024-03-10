from flask import Flask
from flask_restful import Api, Resource, reqparse
from chroma_db import ChromaDB
import logging

app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO, filename='flask.log', filemode="w",
                       format="%(asctime)s %(levelname)s %(message)s")

class Quote(Resource):
    def get(self):
        logging.info("get")

        parser = reqparse.RequestParser()
        parser.add_argument("return_all")
        parser.add_argument("query")
        params = parser.parse_args()

        if params["return_all"] is not None:
            logging.info("get start")
            result = chroma.select_all()
            logging.info("get end")
            return result

        
        response = chroma.query(params["query"])

        if response:
            return response, 200
        
        return "Error: query not found", 404
    
    def post(self):
        logging.info("post")

        parser = reqparse.RequestParser()
        parser.add_argument("chunks")
        parser.add_argument("sources")
        parser.add_argument("document_type")
        parser.add_argument("pages")
        params = parser.parse_args()

        if params["pages"] is not None: 
            pages = eval(params["pages"])
        else:
            pages = None
        
        response = chroma.insert_document(chunks=eval(params["chunks"]), sources=eval(params["sources"]), document_type=params["document_type"], pages=pages)

        if response:
            return f"Error: {response}", 404
        else:
            return 'The document has been added successfully', 200
    
    def put(self):
        logging.info("put")

        parser = reqparse.RequestParser()
        parser.add_argument("old_document_id")
        parser.add_argument("chunks")
        parser.add_argument("sources")
        parser.add_argument("document_type")
        parser.add_argument("pages")
        params = parser.parse_args()

        if params["pages"] is not None: 
            pages = eval(params["pages"])
        else:
            pages = None
        
        response = chroma.upsert_document(old_document_id=int(params['old_document_id']), chunks=eval(params["chunks"]), sources=eval(params["sources"]), document_type=params["document_type"], pages=pages)
        if response:
            return f"Error: {response}", 404
        else:
            return 'The document has been upserted successfully', 200
    
    def delete(self):
        logging.info("delete")

        parser = reqparse.RequestParser()
        parser.add_argument("ids")
        params = parser.parse_args()
        ids = eval(params["ids"])

        response = chroma.delete_documents(ids)
        if response:
            return f'Error: {response}', 404
        else:
            return f"Rows with ids: {ids} is deleted.", 200
    
api.add_resource(Quote, "/")
if __name__ == "__main__":
    logging.info("start run flask")

    chroma = ChromaDB()

    logging.info("crashed")
    app.run(host='0.0.0.0', port = 5000)
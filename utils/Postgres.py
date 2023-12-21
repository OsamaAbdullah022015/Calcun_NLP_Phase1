# class Postgre:
#     def __init__(self, database, user, pdf_name, host="postgres", port="5432", password="1234"):
#         self.pdf_name = pdf_name
#         try:
#             self.conn = psycopg2.connect(database=database, user=user, host=host, port=port, password=password)
#             self.cur = self.conn.cursor()
#             print(f"Connected to database {database} with user name as {user}")
#         except e:
#             print("Unable to connect to database due to error : ", e)
#         self.createTable('SVGTable')
#         self.createTable('JSONTable')
        
#     def execute(self, query):
#         '''
#         Function to execute some query
#         '''
#         try:
#             self.cur.execute(query)
#             print(f"Query '{query}' successfully executed")
#         except e:
#             print("Query could not be executed due to error: ", e)
            
#     def commit(self):
#         '''
#         Function to commit changes
#         '''
#         try:
#             self.conn.commit()
#         except e:
#             print("Could not commit due to error: ", e)
            
#     def close(self):
#         '''
#         Function to close the connection
#         '''
#         try:
#             self.cur.close()
#             self.conn.close()
#         except e:
#             print("Could not close the connection due to error: ", e)

#     def createTable(self, table_name):
#         '''
#         Function to create table if it does not exist already.
#         '''
#         try:
#             if table_name == "SVGTable":
#                 self.cur.execute(f'''CREATE TABLE IF NOT EXISTS {table_name}(
#                 FileName VARCHAR(100) NOT NULL PRIMARY KEY, 
#                 Data BYTEA NOT NULL)''')
#             else:
#                 self.cur.execute(f'''CREATE TABLE IF NOT EXISTS {table_name}(
#                 FileName VARCHAR(100) NOT NULL PRIMARY KEY, 
#                 Data JSONB NOT NULL)''')
#             print("Successfully created table ", table_name)
#         except(Exception, psycopg2.Error) as error:
#             print("Error while creating table", error)
#         finally:
#             self.conn.commit()

#     def write_blob(self,table_name,file_name,data=None):
#         '''
#         Function to store data in data base
#         '''
#         try:
#             if not data:
#                 file_path = "./output/"+file_name
#                 data = open(file_path, 'rb').read()
#                 data = psycopg2.Binary(data)
#             try:
#                 self.cur.execute("INSERT INTO " + table_name + " (filename, data) " + "VALUES(%s,%s)",(self.pdf_name.split('.')[0]+'_'+file_name, data))
#                 self.conn.commit()
#             except (Exception, psycopg2.DatabaseError) as error:
#                 print(f"Error while inserting data in {table_name} table: {error}")
#         except e:
#             print(f"Error while opening the file: {e}")

#     def retrieve_data(self, file_name):
#         '''
#         Funtion to retrieve data from database
#         '''
#         #Declaring name
#         name = self.pdf_name.split('.')[0]+'_'+file_name

#         #Extracting and displaying data from JSONTable
#         exe = self.cur.execute("SELECT * from jsontable;")
#         jsontable = self.cur.fetchall()
#         for file,data in jsontable:
#             if file == name+'.json':
#                 print(data)
#                 break

#         #Extracting and displaying data from svgtable
#         exe = self.cur.execute("SELECT * from svgtable;")
#         svgtable = self.cur.fetchall()
#         for file,data in svgtable:
#             if file == name+'.svg':
#                 open('./output/'+name+'.svg', 'wb').write(data)
#                 display(SVG('./output/'+name+'.svg'))
#                 os.remove('./output/'+name+'.svg')
#                 break        
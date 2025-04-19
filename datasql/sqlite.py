import sqlite3


connection  = sqlite3.connect("pr_report.db")

cursor = connection.cursor()

# tabel_info = """

# create table prreport(pr_id int(20), rquested_by varchar(25), location varchar(25), 
# product_name varchar(25), product_id int(10), 
# approved_by varchar(25), vendor varchar(25), approval_status int(1))


# """

# cursor.execute(tabel_info)

# cursor.execute(''' Insert into prreport values(001,'Thirumurugan','koramangala','apple',01,'siddharth','insfusion',1)''')
# cursor.execute(''' Insert into prreport values(002,'Sagar','Nexus mall','orange',02,'madhushree','ffservices',1)''')
# cursor.execute(''' Insert into prreport values(003,'Thirumurugan','rich street','pineapple',03,'sagar','ananya shelter',1)''')
# cursor.execute(''' Insert into prreport values(004,'madhushree','koramangala','watermelon',04,'Thirumurugan','devalt',1)''')
# cursor.execute(''' Insert into prreport values(005,'krishna','madiwala','grapes',05,'','zepto',0)''')
# cursor.execute(''' Insert into prreport values(006,'Thirumurugan','koram','grapes',05,'','gssprojects',0)''')
# cursor.execute(''' Insert into prreport values(007,'Krishna','bommanahalli','apple',06,'','insfusion',0)''')


print("Records are inserted!!")

data = cursor.execute('''select * from prreport where approval_status = 1''')

for row in data:
    print(row)

connection.commit()
connection.close()
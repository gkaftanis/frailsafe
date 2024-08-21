import arango

arango_db_name='frailSafeProject'
arango_username='frailsafe_db_user'
arango_password='frailsafe_gk@1'

def database_setup(password,arango_nodes,username='root'):
    client = arango.ArangoClient(hosts=arango_nodes)
    system_db = client.db('_system',
                          username='root',
                          password=password)

    print(system_db.databases())

    if not system_db.has_database(arango_db_name):
        users=[{
                'username':arango_username,
                'password':arango_password,
                'active':True
               }]
        system_db.create_database(
                name=arango_db_name,
                users=users)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Database Setup')
    parser.add_argument('--root_password',type=str,
        		help='specify the root password',
                        required=True)
    parser.add_argument('--arango_nodes',type=str,
                        help='comma seperated ip:port:hostname:port',
                        required=True)

    args=parser.parse_args()
    arango_nodes=args.arango_nodes.split()

    database_setup(password=args.root_password,
                   arango_nodes=arango_nodes)

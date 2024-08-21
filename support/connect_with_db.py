import arango
import settings


class DBConnection():
    def __init__(self):
        self.active_collections = {}
        self.client = arango.ArangoClient(hosts=settings.arango_nodes,request_timeout = 120)
        self.arango_db = self.client.db(settings.arango_db_name,
                                        username=settings.arango_username,
                                        password=settings.arango_password)

    def get_collection(self, obj):
        if obj.__class__ != type:
            obj = obj.__class__
        collection_name = obj.__name__
        if collection_name in self.active_collections:
            return self.active_collections[collection_name]
        if self.arango_db.has_collection(collection_name):
            collection = self.arango_db.collection(collection_name)
        else:
            collection = self.arango_db.create_collection(collection_name)
        self.active_collections[collection_name] = collection
        return self.active_collections[collection_name]


collection_registry = DBConnection()


class ConnectwithDB(object):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_key_value(cls):
        raise Exception('implement me')
        return attr_name

    def to_dict(self):
        self._key = getattr(self, self.get_key_value())
        return dict(self.__dict__)

    def to_class(self, dictionary):
        self.__dict__.update(dictionary)
        setattr(self, self._key, self.get_key_value())

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, dictionary):
        self.to_class(dictionary)

    def setstate(self, dictionary):
        self.__dict__.update(dictionary)
        # probably redundant if we use it over db
        setattr(self, self._key, self.get_key_value())

    def db_write(self):
        db_collection = collection_registry.get_collection(self)
        db_collection.insert(self.to_dict(),
                             overwrite=True)

    def db_read(self, key):
        db_collection = collection_registry.get_collection(self)
        data_dict = db_collection.get(key)
        return data_dict


def batch_db_read(theclass, list_of_keys_or_ids=None):  # None for all
    db_collection = collection_registry.get_collection(theclass)

    if list_of_keys_or_ids is None:
        cursor = db_collection.all()
        for c in cursor:
            obj = theclass()
            obj.setstate(c)
            yield obj
    else:
        dicts = db_collection.get_many(list_of_keys_or_ids)
        for d in dicts:
            obj = theclass()
            obj.setstate(d)
            yield obj


def batch_db_write(list_of_objects,
                   disk_sync=False):
    classes = {}
    for o in list_of_objects:
        c = o.__class__
        if c not in classes:
            classes[c] = []
        classes[c].append(o)

    for c, objs in classes.items():
        for objs_chunk in get_chunks(objs, 10000):
            db_collection = collection_registry.get_collection(c)
            db_collection.insert_many([cleaner(o.getstate()) for o in objs_chunk],
                                      overwrite=True,
                                      sync=disk_sync)


def do_aql(aql):
    db = collection_registry.arango_db
    cursor = db.aql.execute(aql,
                            batch_size=10000)
    return cursor


def read_collection_keys(collection):
    aql = " for k in " + collection + \
          " return k._key"
    dicts = do_aql(aql)
    doc_ids = []
    for d in dicts:
        doc_ids.append(d)
    return doc_ids


def inner_join_collections(theclass1, theclass2, key1='_key', key2='_key'):
    # returns a dictionary
    collection1 = collection_registry.get_collection(theclass1)
    collection2 = collection_registry.get_collection(theclass2)
    aql = "for doc1 in " + collection1.name + \
          " for doc2 in " + collection2.name + \
          " filter doc1." + key1 + "==doc2." + key2 + \
          " return MERGE(doc1, doc2)"
    return do_aql(aql)


def receive_last_timestamp(theclass, timestamp_name='timestamp'):
    db_collection = collection_registry.get_collection(theclass)

    aql = "FOR k IN " + db_collection.name + \
          " FILTER k." + timestamp_name + \
          " SORT k." + timestamp_name + " DESC " + \
          " LIMIT 1 " + \
          " RETURN k." + timestamp_name
    # print(aql)
    dicts = do_aql(aql)
    doc_ids = []
    for d in dicts:
        doc_ids.append(d)
    try:
        return doc_ids[0]
    except:
        return 0


def receive_first_timestamp(theclass, timestamp_name='timestamp'):
    db_collection = collection_registry.get_collection(theclass)

    aql = "FOR k IN " + db_collection.name + \
          " FILTER k." + timestamp_name + \
          " SORT k." + timestamp_name + "  " + \
          " LIMIT 1 " + \
          " RETURN k." + timestamp_name
    print(aql)
    dicts = do_aql(aql)
    doc_ids = []
    for d in dicts:
        doc_ids.append(d)
    try:
        return doc_ids[0]
    except:
        return 0


def receive_participants_with_one_status(theclass):
    collection = collection_registry.get_collection(theclass)
    aql = " FOR doc IN " + collection.name + " " \
                                             "FILTER LENGTH(doc.frailty_status) == 1" +" "+ \
          "RETURN doc._key"

    dicts = do_aql(aql)
    doc_ids = []
    for d in dicts:
        doc_ids.append(d)
    return doc_ids


def extract_sorted_collection(theclass,
                              ascending=True,
                              limit=-1,
                              low_timestamp_threshold=None,
                              high_timestamp_threshold=None,
                              timestamp_variable='timestamp',
                              return_cursor=False,
                              participant_id_filter=None,
                              activity_class_filter=None
                              ):
    db_collection = collection_registry.get_collection(theclass)

    aql = "for k in " + db_collection.name

    if low_timestamp_threshold is not None or high_timestamp_threshold is not None or (participant_id_filter is not None):
        aql = aql + " filter "

    if low_timestamp_threshold is not None:
        aql = aql + "  TO_NUMBER(k." + timestamp_variable + ") >" + str(low_timestamp_threshold)

    if high_timestamp_threshold is not None:
        if low_timestamp_threshold is not None:
            aql = aql + " and "
        aql = aql + "  TO_NUMBER(k." + timestamp_variable + ") <" + str(high_timestamp_threshold)

    if participant_id_filter is not None:
        if high_timestamp_threshold is not None or low_timestamp_threshold is not None:
            aql = aql + " and "
        aql = aql + " k.participant_id=='"+str(participant_id_filter)+"' "

    if activity_class_filter is not None:
        aql = aql + " and k.activity_class2=='" + str(activity_class_filter) + "' "

    aql = aql + " SORT k." + timestamp_variable + " "
    if not ascending:
        aql = aql + "DESC "

    if limit != -1:
        aql = aql + "LIMIT " + str(limit)

    aql = aql + " " + "return k"

    dicts = do_aql(aql)

    if return_cursor:
        return dicts

    for d in dicts:
        obj = theclass()
        obj.setstate(d)
        yield obj



def count_collection_entries_with_timestamp(theclass, filter_timestamp=0):
    db_collection = collection_registry.get_collection(theclass)
    aql = "for k in " + db_collection.name

    if filter_timestamp:
        aql = aql + ' filter TO_NUMBER(k.timestamp)>' + str(filter_timestamp)

    aql = aql + ' RETURN 1'
    aql = 'RETURN COUNT ( ' + aql + ' )'

    dicts = do_aql(aql)
    for d in dicts:
        return d

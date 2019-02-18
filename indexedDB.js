class IndexedDB {

    constructor() {
        this.indexedDB = window.indexedDB || window.mozIndexedDB || window.webkitIndexedDB || window.msIndexedDB || window.shimIndexedDB;
        this.open=null;
        this.db=null;
    }

    openDB(){
        return new Promise((resolve, reject)=>{
            this.open=this.indexedDB.open("MyDatabase", 1);
            var _this=this;
            this.open.onupgradeneeded = function() {
                var db = _this.open.result;
                var store = db.createObjectStore("tonyModel", {keyPath: "name"});
            };
            this.open.onsuccess = function () {
                _this.db=_this.open.result;
                resolve()
            }
            this.open.onerror = function () {
                reject()
            }
        })
    }

    store(name,val){
      if(this.db==null)
          throw "DB is not yet opened";
      else{
          var _this=this;
          var tx = this.db.transaction("tonyModel", "readwrite");
          var store = tx.objectStore("tonyModel");
          store.put({name: name,value:val});
          tx.oncomplete = function () {
              _this.db.close();
          };
      }
    }

    getValue(name){
        return new Promise((resolve, reject)=>{
            if(this.db==null)
                throw "DB is not yet opened";
            else{
                var _this=this;
                var tx = this.db.transaction("tonyModel", "readwrite");
                var store = tx.objectStore("tonyModel");
                var result=store.get(name);
                tx.oncomplete = function () {
                    _this.db.close();
                };
                result.onsuccess = function () {
                    if(result.result)
                        resolve(result.result.value);
                    else
                        resolve();    
                };
                result.onerror = function () {
                    reject("not found");
                };
            }
        })
    }

  dosmt () {
        // Start a new transaction
        var db = open.result;
        var tx = db.transaction("MyObjectStore", "readwrite");
        var store = tx.objectStore("MyObjectStore");
        var index = store.index("NameIndex");

        // Add some data
        store.put({id: 12345, name: {first: "John", last: "Doe"}, age: 42});
        store.put({id: 67890, name: {first: "Bob", last: "Smith"}, age: 35});

        // Query the data
        var getJohn = store.get(12345);
        var getBob = index.get(["Smith", "Bob"]);

        getJohn.onsuccess = function () {
            console.log(getJohn.result.name.first);  // => "John"
        };

        getBob.onsuccess = function () {
            console.log(getBob.result.name.first);   // => "Bob"
        };

        // Close the db when the transaction is done
        tx.oncomplete = function () {
            db.close();
        };
    }


}
export { IndexedDB };
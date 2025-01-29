import { Request, Response } from 'express';
import * as mqtt from 'mqtt';
let express = require('express')
import { MongoClient, ObjectId, WithId, Document } from 'mongodb';
import { json_replacer, Thing, ThingUpdate } from './common';
import { tstwin_client } from './clients';

const transformDocumentToThing = (doc: WithId<Document>): Thing => {
    return {
        _id: doc._id.toString(),
        name: doc.name,
        description: doc.description,
        type: doc.type,
        metadata: doc.metadata,
        state: doc.state,
        onCreate: new Function('return ' + doc.onCreate)(),
        onUpdate: new Function('return ' + doc.onUpdate)(),
        onDelete: new Function('return ' + doc.onDelete)()
    };
};

const url = 'mongodb://localhost:27017';
const dbName = 'thingsDB';

let client = await MongoClient.connect(url)
const db = client.db(dbName);

const mongoclient = {
    async createThing(thing: Thing) {
        const collection = db.collection('things');
        let t2: any = {
            ...thing
        };
        t2._id = new ObjectId(thing._id.padStart(24, '0'));
        await collection.insertOne(t2);
    },

    async updateThing(updatedThing: Thing) {
        const collection = db.collection('things');
        let _id = new ObjectId(updatedThing._id.padStart(24, '0'));
        let t: any = updatedThing
        delete t._id
        await collection.updateOne({ _id }, { $set: t });
    },

    async deleteThing(_id: string) {
        const collection = db.collection('things');
        const result = await collection.deleteOne({ _id: new ObjectId(_id.padStart(24, '0')) });
        return result.deletedCount > 0;
    },

    async getThing(id: string): Promise<Thing | null> {
        const collection = db.collection('things');
        let _id = new ObjectId(id.padStart(24, '0'));
        let thing = await collection.findOne({ _id });
        if (!thing) { return null }
        else {
            return transformDocumentToThing(thing);
        }
    },

    async listThings() {
        const collection = db.collection('things');
        return await collection.find({}).toArray();
    }
}

//const topics = ['echo', 'create/thing', 'update/thing', 'delete/thing', 'get/thing', 'list/things'];

const handler = {
    mongoclient,
    db,
    libraries: {
        baseURL: 'http://localhost:3000',
        client: tstwin_client,
        json_replacer,
        mqtt,
    },

    create_thing: async function (thing: Thing) {
        if (await mongoclient.getThing(thing._id)) {
            console.error(`Thing already exists: ${thing._id}`);
            return {
                success: false,
                error: `Thing already exists: ${thing._id}`
            }
        }
    
        if (thing.onCreate) {
            const onCreate = new Function('return ' + thing.onCreate)();
            onCreate(thing, this.libraries);
        }
        await mongoclient.createThing(thing);
        return {
            success: true,
        }
    },
    
    update_thing: async function (updatedThing: ThingUpdate) {
        const oldThing = await mongoclient.getThing(updatedThing._id);
        if (!oldThing) {
            console.error(`Thing not found: ${updatedThing._id}`);
            return {
                success: false,
                error: `Thing not found: ${updatedThing._id}`
            }
        }
    
        let updated: Thing = { ...oldThing, ...updatedThing };
        if (oldThing.onUpdate) {
            const onUpdate = new Function('return ' + oldThing.onUpdate)();
            onUpdate(oldThing, updated, this.libraries);
        }
    
        await mongoclient.updateThing(updated);
        return {
            success: true,
        }
    },
    
    delete_thing: async function (id: string) {
        const thingToDelete = await mongoclient.getThing(id);
        if (!thingToDelete) {
            console.error(`Thing not found: ${id}`);
            return {
                success: false,
                error: `Thing not found: ${id}`
            }
        }
    
        if (thingToDelete.onDelete) {
            const onDelete = new Function('return ' + thingToDelete.onDelete)();
            onDelete(thingToDelete, this.libraries);
        }
    
    
        if (await mongoclient.deleteThing(id)) {
            console.log('Thing deleted:', id);
            return {
                success: true,
            }
        } else {
            return {
                success: false,
                error: `Thing not found: ${id}`
            }
        }
    },
    
    get_thing: async function (idToGet: string) {
        const thingToGet = await mongoclient.getThing(idToGet);
        if (!thingToGet) {
            console.error(`Thing not found: ${idToGet}`);
            return {
                success: false,
                error: `Thing not found: ${idToGet}`
            }
        }
        return {
            success: true,
            data: thingToGet
        }
    }
}




if (require.main === module) {
    (async () => {
        const app = express();
        app.use(express.json());

        app.get('/things', async (req: Request, res: Response) => {
            res.json(mongoclient.listThings());
        });

        app.get('/thing/:id', async (req: Request, res: Response) => {
            const { success, data, error } = await handler.get_thing(req.params.id);
            if (success) {
                res.status(200).send(JSON.stringify(data, json_replacer));
            } else {
                res.status(404).json({ error });
            }
        });

        app.post('/thing', async (req: Request, res: Response) => {
            const { success, error } = await handler.create_thing(req.body);
            if (success) {
                res.status(200).send();
            } else {
                res.status(400).json({ error });
            }
        })

        app.put('/thing/:id', async (req: Request, res: Response) => {
            const { success, error } = await handler.update_thing(req.body);
            if (success) {
                res.status(200).send();
            } else {
                res.status(404).json({ error });
            }
        })

        app.delete('/thing/:id', async (req: Request, res: Response) => {
            const { success, error } = await handler.delete_thing(req.params.id);
            if (success) {
                res.status(200).send();
            } else {
                res.status(404).json({ error });
            }
        })

        app.listen(3000, () => {
            console.log('Server listening on port 3000');
        });
    })();
}
/*
function dispatcher(topic: string, message: string): { success: boolean, data?: any, error?: string } {
    console.log('Received message:', topic, message);
    switch (topic) {
        case "echo":
            return echo(message);
        case "create/thing":
            return create_thing(message);
        case "update/thing":
            return update_thing(message);
        case "delete/thing":
            return delete_thing(message);
        //case "get/thing":
        //    return get_thing(message);
        default:
            console.error(`Unknown topic: ${topic}`);
            return {
                success: false,
                data: `Unknown topic: ${topic}`
            }
    }
}


const client = mqtt.connect('mqtt://172.17.0.2:1883', {
    clientId: 'test-client',
    keepalive: 5
});

client.on('connect', () => {
    client.subscribe(topics, { qos: 0 }, (err) => {
        if (err) {
            console.error('Subscription error:', err);
        }
    });
});

client.on('message', (topic, message) => {
    try {
        const { success, data, error } = dispatcher(topic, message.toString());
        if (success) {
            if (data) {
                client.publish('response', JSON.stringify(data), { qos: 0 });
            }
        } else {
            client.publish('response', JSON.stringify({ error }), { qos: 0 });
        }
    } catch (e) {
        console.error('Error parsing message:', e);
    }
});

client.on('error', (err) => {
    console.error('Error =', err);
});*/




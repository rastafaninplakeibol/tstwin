import { Thing, ThingUpdate, json_replacer } from "./common";

const baseURL = 'http://localhost:3000';

export const tstwin_client = {
    baseURL: baseURL,
    json_replacer,

    createThing: async function (thing: Thing) {
        const create_res = await fetch(`${this.baseURL}/thing`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(thing, this.json_replacer)
        });
        if (create_res.status !== 200) {
            throw new Error('Failed to create thing');
        } else {
            console.log('Created thing:', thing);
        }
    },

    getThing: async function (id: string) { 
        const get_res = await fetch(`${this.baseURL}/thing/${id}`);
        let thing: Thing = await get_res.json();

        thing.onCreate = new Function('return ' + thing.onCreate)(),
        thing.onUpdate = new Function('return ' + thing.onUpdate)(),
        thing.onDelete = new Function('return ' + thing.onDelete)()

        return thing;
    },

    updateThing: async function (updatedThing: ThingUpdate) {
        const updateResponse = await fetch(`${this.baseURL}/thing/${updatedThing._id}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updatedThing, this.json_replacer)
        });
        if (updateResponse.status !== 200) {
            throw new Error('Failed to update thing');
        }
    },

    deleteThing: async function (id: string) {
        const deleteResponse = await fetch(`${this.baseURL}/thing/${id}`, {
            method: 'DELETE'
        });
        if (deleteResponse.status !== 200) {
            throw new Error('Failed to delete thing');
        } else {
            console.log('Deleted thing: ' + id);
        }
    }
};

async function simulate_temperature_change() {
    let { temperature } = (await tstwin_client.getThing('2')).state;

    while (true) {
        let caldaia = await tstwin_client.getThing('1');
        
        if(caldaia.state.active) {
            temperature += Math.random() * 2;   
        } else {
            temperature -= Math.random() * 2;
        }
        
        let update: ThingUpdate = {
            _id: '2',
            state: {
                temperature
            }
        }
        console.log('Updating temperature to:', temperature);
        await tstwin_client.updateThing(update);
        await sleep(5000);
    }
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

if (require.main === module) {
    const thing: Thing = {
        _id: '1',
        name: 'CaldaiaCucina',
        description: 'Una caldaia di test',
        type: 'test',
        metadata: {
            position: 'cucina',
        },
        state: {
            active: false,
        },
        onCreate: (thing: Thing, libraries: any) => {
            console.log('Caldaia created');
        },

        onUpdate: (oldThing: Thing, updatedThing: Thing, libraries: any) => {
            if (updatedThing.state.active) {
                console.log('Caldaia is now on');
                const mqtt = libraries.mqtt;
                let client = mqtt.connect('mqtt://172.17.0.2:1883')
                client.publish('caldaia', '{ "active": true }'); 

            } else {
                console.log('Caldaia is now off');
                const mqtt = libraries.mqtt;
                let client = mqtt.connect('mqtt://172.17.0.2:1883')
                client.publish('caldaia', '{ "active": false }'); 
            }
        },
    };

    const thing2: Thing = {
        _id: '2',
        name: 'Termostato',
        description: 'Un termostato di test',
        type: 'test',
        metadata: {
            position: 'soggiorno',
        },
        state: {
            temperature: 20
        },
        onCreate: (thing: Thing, libraries: any) => {
            console.log('Termostato created');
        },

        onUpdate: async (oldThing: Thing, updatedThing: Thing, libraries: any) => {
            const client = libraries.client;
            
            let caldaia = await client.getThing('1');
            if (updatedThing.state.temperature > 25 && caldaia.state.active) {
                console.log('Temperature is too high');
                let update: ThingUpdate = {
                    _id: caldaia._id,
                    state: {
                        ...caldaia.state,
                        active: false,
                    }
                }
                client.updateThing(update);
            } else if (updatedThing.state.temperature < 15 && !caldaia.state.active) {
                console.log('Temperature is too low');
                let update: ThingUpdate = {
                    _id: caldaia._id,
                    state: {
                        ...caldaia.state,
                        active: true,
                    }
                }
                client.updateThing(update);
            }
        },
    };

    (async () => {
        try {
            //await tstwin_client.deleteThing('1');
            //await tstwin_client.deleteThing('2');
            //await tstwin_client.createThing(thing);
            //await tstwin_client.createThing(thing2);
            //await tstwin_client.updateThing({ _id: '1', onUpdate: thing.onUpdate });
            //await tstwin_client.updateThing({ _id: '2', onUpdate: thing2.onUpdate });
            //await sleep(2000);
            console.log(await tstwin_client.getThing('1'));
            console.log(await tstwin_client.getThing('2'));
            await sleep(2000);
            //await simulate_temperature_change();
        } catch (error) {
            console.error(error);
        }
    })();
}



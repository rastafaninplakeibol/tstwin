export interface Thing {
    _id: string;
    name: string;
    description: string;
    type: string;
    metadata: any;
    state: any;

    onCreate?: (thing: Thing, libraries: any) => void;
    onUpdate?: (oldThing: Thing, newThing: Thing, libraries: any) => void;
    onDelete?: (thing: Thing, libraries: any) => void;
}

export interface ThingUpdate {
    _id: string;
    name?: string;
    description?: string;
    type?: string;
    metadata?: any;
    state?: any;

    onCreate?: (thing: Thing, libraries: any) => void;
    onUpdate?: (oldThing: Thing, newThing: Thing, libraries: any) => void;
    onDelete?: (thing: Thing, libraries: any) => void;
}


export const json_replacer = (key: string, value: any) => {
    if (typeof value === 'function') {
        return value.toString();
    }
    return value;
};
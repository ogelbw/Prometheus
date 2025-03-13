import express from 'express';
import mineflayer from 'mineflayer';
import {pathfinder, Movements, goals} from 'mineflayer-pathfinder';

const app = express();
const port = 3000;
const bot_client = mineflayer.createBot({
    host: 'localhost',
    username: 'Bot',
    port: 25565,
    auth: 'offline'
})

bot_client.once('spawn', () => {
    bot_client.chat('Hello world!')
})

bot_client.on('playerLeft', _ => {
    // => ['bot left the game']
    console.log('bot has just left!')
})

bot_client.on('playerJoined', player => {
    console.log(`${player.username} has just joined!`) // should only run once
})

bot_client.on('chat', (username, msg) => {
    console.log(`(Chat)[${username}]: ${msg}`)
})

app.use(express.json());

interface goto { radius: number, target: {x: number, y: number} }
app.post('/player/nav/goto', (req, res) => {
    const { radius, target }: goto = req.body;
    res.json({ message: 'Going to destination', data: req.body });
    console.log(radius)
});

interface follow { username: string }
app.post('/player/nav/follow', (req, res) => {
    res.json({ message: 'Following target', data: req.body });
    const { username } : follow = req.body 
});

// Player actions
app.post('/player/action/attack', (req, res) => {
    res.json({ message: 'Attacking target', data: req.body });
});

app.post('/player/action/punch', (req, res) => {
    res.json({ message: 'Punching target', data: req.body });
});

app.post('/player/action/say', (req, res) => {
    bot_client.chat(req.body.value)
    res.json({message: `Saying: ${req.body.value}`})
});

app.post('/player/action/interact_block', (req, res) => {
    res.json({ message: 'Interacting with block', data: req.body });
});

app.post('/player/action/interact_entity', (req, res) => {
    res.json({ message: 'Interacting with entity', data: req.body });
});

// Player inventory
app.post('/player/inventory/hotbar_select', (req, res) => {
    res.json({ message: 'Selecting hotbar slot', data: req.body });
});

app.get('/player/inventory/get', (req, res) => {
    res.json({ message: 'Fetching inventory contents' });
});

app.post('/player/inventory/move_item', (req, res) => {
    res.json({ message: 'Moving item in inventory', data: req.body });
});

// Player look
app.post('/player/look_at', (req, res) => {
    res.json({ message: 'Looking at target', data: req.body });
});

// World interactions
app.get('/world/get_entities', (req, res) => {
    res.json({ message: 'Fetching entities in the world' });
});

app.get('/world/find_blocks', (req, res) => {
    res.json({ message: 'Finding blocks of a certain type', data: req.body });
});

app.post('/world/register_poi', (req, res) => {
    res.json({ message: 'Registering point of interest', data: req.body });
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});

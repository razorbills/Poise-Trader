import { io } from 'socket.io-client'

// Connect to same-origin Socket.IO (proxied in dev, same host in prod)
const socket = io({
  transports: ['websocket', 'polling'],
})

export default socket

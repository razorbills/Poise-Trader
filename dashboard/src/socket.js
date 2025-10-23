import io from 'socket.io-client'

const BACKEND_URL = import.meta.env.DEV 
  ? 'http://localhost:5000' 
  : window.location.origin

const socket = io(BACKEND_URL, {
  reconnection: true,
  reconnectionDelay: 500,
  reconnectionAttempts: Infinity,
  transports: ['websocket', 'polling']
})

socket.on('connect', () => {
  console.log('✅ Connected to backend')
})

socket.on('disconnect', () => {
  console.log('❌ Disconnected from backend')
})

export default socket

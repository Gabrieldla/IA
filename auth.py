"""
Sistema de autenticación local simple
"""
import json
from pathlib import Path
import hashlib

AUTH_FILE = Path("users.json")

def hash_password(password):
    """Genera hash SHA256 de la contraseña"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Carga usuarios del archivo JSON"""
    if not AUTH_FILE.exists():
        return {}
    with open(AUTH_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    """Guarda usuarios en archivo JSON"""
    with open(AUTH_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(username, password, name=None, email=None):
    """Registra un nuevo usuario"""
    users = load_users()
    
    if username in users:
        return False, "El usuario ya existe"
    
    if len(username) < 3:
        return False, "El usuario debe tener al menos 3 caracteres"
    
    if len(password) < 4:
        return False, "La contraseña debe tener al menos 4 caracteres"
    
    if email:
        # Validación básica de email
        if '@' not in email or '.' not in email:
            return False, "El correo electrónico no es válido"
    
    users[username] = {
        'password': hash_password(password),
        'name': name or username,
        'email': email or ''
    }
    save_users(users)
    return True, "Usuario registrado exitosamente"

def login_user(username, password):
    """Valida credenciales de usuario"""
    users = load_users()
    
    if username not in users:
        return False, "Usuario no encontrado"
    
    if users[username]['password'] != hash_password(password):
        return False, "Contraseña incorrecta"
    
    return True, "Login exitoso"

def get_user_info(username):
    """Obtiene información del usuario"""
    users = load_users()
    return users.get(username, None)

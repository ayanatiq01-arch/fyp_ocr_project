// login_screen.dart
//
// First page: sign in, or create an account. Accounts are kept on this
// phone (see auth_service.dart); "Stay signed in" skips this page next time.

import 'package:flutter/material.dart';

import 'auth_service.dart';
import 'theme.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key, required this.auth});

  final AuthService auth;

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> with SingleTickerProviderStateMixin {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _email = TextEditingController();
  final _password = TextEditingController();
  final _confirm = TextEditingController();
  late final AnimationController _intro =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))..forward();

  bool _signUp = false;
  bool _remember = true;
  bool _showPassword = false;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _intro.dispose();
    _name.dispose();
    _email.dispose();
    _password.dispose();
    _confirm.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() => _error = null);
    if (!(_form.currentState?.validate() ?? false)) return;
    setState(() => _busy = true);
    final error = _signUp
        ? await widget.auth.signUp(
            name: _name.text, email: _email.text, password: _password.text, remember: _remember)
        : await widget.auth.signIn(email: _email.text, password: _password.text, remember: _remember);
    if (!mounted) return;
    setState(() {
      _busy = false;
      _error = error;
    });
  }

  void _toggleMode() => setState(() {
        _signUp = !_signUp;
        _error = null;
        _form.currentState?.reset();
      });

  @override
  Widget build(BuildContext context) {
    final fade = CurvedAnimation(parent: _intro, curve: Curves.easeOutCubic);
    return Scaffold(
      body: HarfBackground(
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(24, 24, 24, 24),
              child: FadeTransition(
                opacity: fade,
                child: SlideTransition(
                  position: Tween(begin: const Offset(0, 0.06), end: Offset.zero).animate(fade),
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 440),
                    child: Column(
                      children: [
                        const HarfLogo(size: 84, showTagline: false),
                        const SizedBox(height: 6),
                        Text(_signUp ? 'Create your account' : 'Sign in to continue',
                            style: Theme.of(context).textTheme.bodyMedium),
                        const SizedBox(height: 24),
                        _buildCard(),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCard() {
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 22, 20, 16),
        child: Form(
          key: _form,
          child: AutofillGroup(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                AnimatedSize(
                  duration: const Duration(milliseconds: 250),
                  child: _signUp
                      ? Padding(
                          padding: const EdgeInsets.only(bottom: 14),
                          child: TextFormField(
                            controller: _name,
                            textCapitalization: TextCapitalization.words,
                            textInputAction: TextInputAction.next,
                            autofillHints: const [AutofillHints.name],
                            decoration: _field('Full name', Icons.person_outline),
                            validator: (v) => AuthService.validateName(v ?? ''),
                          ),
                        )
                      : const SizedBox(width: double.infinity),
                ),
                TextFormField(
                  controller: _email,
                  keyboardType: TextInputType.emailAddress,
                  textInputAction: TextInputAction.next,
                  autocorrect: false,
                  autofillHints: const [AutofillHints.email],
                  decoration: _field('Email', Icons.alternate_email),
                  validator: (v) => AuthService.validateEmail(v ?? ''),
                ),
                const SizedBox(height: 14),
                TextFormField(
                  controller: _password,
                  obscureText: !_showPassword,
                  textInputAction: _signUp ? TextInputAction.next : TextInputAction.done,
                  autofillHints: [_signUp ? AutofillHints.newPassword : AutofillHints.password],
                  decoration: _field('Password', Icons.lock_outline).copyWith(
                    suffixIcon: IconButton(
                      tooltip: _showPassword ? 'Hide password' : 'Show password',
                      icon: Icon(_showPassword ? Icons.visibility_off : Icons.visibility),
                      onPressed: () => setState(() => _showPassword = !_showPassword),
                    ),
                    helperText: _signUp ? 'At least 6 characters, with a number' : null,
                  ),
                  validator: (v) => _signUp
                      ? AuthService.validatePassword(v ?? '')
                      : ((v ?? '').isEmpty ? 'Please enter your password' : null),
                  onFieldSubmitted: (_) => _signUp ? null : _submit(),
                ),
                AnimatedSize(
                  duration: const Duration(milliseconds: 250),
                  child: _signUp
                      ? Padding(
                          padding: const EdgeInsets.only(top: 14),
                          child: TextFormField(
                            controller: _confirm,
                            obscureText: !_showPassword,
                            textInputAction: TextInputAction.done,
                            decoration: _field('Confirm password', Icons.lock_reset),
                            validator: (v) => v != _password.text ? 'Passwords do not match' : null,
                            onFieldSubmitted: (_) => _submit(),
                          ),
                        )
                      : const SizedBox(width: double.infinity),
                ),
                const SizedBox(height: 6),
                CheckboxListTile(
                  value: _remember,
                  onChanged: (v) => setState(() => _remember = v ?? true),
                  contentPadding: EdgeInsets.zero,
                  controlAffinity: ListTileControlAffinity.leading,
                  activeColor: HarfColors.gold,
                  checkColor: HarfColors.navy,
                  title: const Text('Stay signed in'),
                ),
                if (_error != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: Text(_error!,
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: Colors.redAccent, fontWeight: FontWeight.w600)),
                  ),
                FilledButton(
                  onPressed: _busy ? null : _submit,
                  style: FilledButton.styleFrom(minimumSize: const Size(0, 52)),
                  child: _busy
                      ? const SizedBox(
                          width: 22,
                          height: 22,
                          child: CircularProgressIndicator(strokeWidth: 2.5, color: HarfColors.navy))
                      : Text(_signUp ? 'Create account' : 'Sign in'),
                ),
                const SizedBox(height: 8),
                TextButton(
                  onPressed: _busy ? null : _toggleMode,
                  child: Text(_signUp
                      ? 'Already have an account? Sign in'
                      : 'New to HarfScan? Create an account'),
                ),
                if (!_signUp)
                  TextButton(
                    onPressed: _busy ? null : _forgotPassword,
                    child: Text('Forgot password?',
                        style: TextStyle(color: HarfColors.ink.withValues(alpha: 0.7))),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _forgotPassword() => showDialog<void>(
        context: context,
        builder: (ctx) => AlertDialog(
          backgroundColor: HarfColors.slate,
          title: const Text('Forgot password'),
          content: const Text(
              'Accounts are stored only on this phone, so the password cannot be sent by email. '
              'Create a new account, or sign in with the password you chose.'),
          actions: [TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('OK'))],
        ),
      );

  InputDecoration _field(String label, IconData icon) => InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon),
        border: const OutlineInputBorder(),
        focusedBorder:
            const OutlineInputBorder(borderSide: BorderSide(color: HarfColors.gold, width: 1.5)),
      );
}
